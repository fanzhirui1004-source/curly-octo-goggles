// Sliver removal for a sheet-solid tetrahedral mesh with the boundary surface locked.
//
// Every sliver the pipeline produces has three vertices on a port cap triangle and one interior vertex almost in
// that plane (sliver_anatomy, 2026-09-05: five meshes, zero slivers with four surface vertices).  Such a sliver is
// removed by moving or reconnecting the INTERIOR vertex only, so the boundary triangulation -- the carrier-conforming
// caps the fixed port interpolates on -- never changes.  CGAL's tetrahedral remeshing with remesh_boundaries(false)
// does exactly that: split / collapse / flip of interior edges and facets and smoothing of interior vertices,
// driven by the minimum dihedral angle.
//
// Fail closed: if any boundary facet or boundary vertex differs after remeshing, exit code 3 and no output mesh.
//
//   pred777h_sliver_remesh --input in.mesh --output out.mesh --target-edge-length L
//                          [--iterations N] [--report out.json]
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Tetrahedral_remeshing/Remeshing_triangulation_3.h>
#include <CGAL/tetrahedral_remeshing.h>
#include <CGAL/IO/File_medit.h>
#include <CGAL/Mesh_3/tet_soup_to_c3t3.h>
#include <sstream>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Tr = CGAL::Tetrahedral_remeshing::Remeshing_triangulation_3<K>;
using Point = K::Point_3;
using Vertex_handle = Tr::Vertex_handle;
using Cell_handle = Tr::Cell_handle;
using P3 = std::array<double, 3>;
using Facet_key = std::array<P3, 3>;

struct Options {
  std::string input, output, report;
  double target_edge_length = 0.0;
  unsigned iterations = 3;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string k(argv[i]);
    auto next = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("missing value for " + k); return argv[i]; };
    if (k == "--input") o.input = next();
    else if (k == "--output") o.output = next();
    else if (k == "--report") o.report = next();
    else if (k == "--target-edge-length") o.target_edge_length = std::stod(next());
    else if (k == "--iterations") o.iterations = static_cast<unsigned>(std::stoul(next()));
    else throw std::runtime_error("unknown argument " + k);
  }
  if (o.input.empty() || o.output.empty() || o.target_edge_length <= 0.0)
    throw std::runtime_error("usage: --input in.mesh --output out.mesh --target-edge-length L [--iterations N] [--report r.json]");
  return o;
}

P3 p3(const Point& p) { return {p.x(), p.y(), p.z()}; }

bool is_domain_cell(const Tr& tr, Cell_handle c) { return !tr.is_infinite(c) && c->subdomain_index() != 0; }

// Boundary facets of the domain as sorted point triples: the exact object the fixed port interpolates on.
std::set<Facet_key> boundary_facets(const Tr& tr) {
  std::set<Facet_key> out;
  for (auto c = tr.finite_cells_begin(); c != tr.finite_cells_end(); ++c) {
    if (!is_domain_cell(tr, c)) continue;
    for (int i = 0; i < 4; ++i) {
      Cell_handle n = c->neighbor(i);
      if (is_domain_cell(tr, n)) continue;
      Facet_key f;
      int k = 0;
      for (int j = 0; j < 4; ++j) if (j != i) f[k++] = p3(c->vertex(j)->point());
      std::sort(f.begin(), f.end());
      out.insert(f);
    }
  }
  return out;
}

struct Quality { double min_dihedral_deg = 180.0; std::size_t below_5 = 0, below_1 = 0, tets = 0; double volume = 0.0; };

double min_dihedral_deg(const std::array<Point, 4>& p) {
  std::array<K::Vector_3, 4> n;
  static const int faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
  for (int k = 0; k < 4; ++k) {
    const Point &a = p[faces[k][0]], &b = p[faces[k][1]], &c = p[faces[k][2]];
    K::Vector_3 v = CGAL::cross_product(b - a, c - a);
    double len = std::sqrt(v.squared_length());
    if (len <= 0.0) return 0.0;
    v = v / len;
    if (v * (p[k] - a) > 0.0) v = -v;           // outward: away from the opposite vertex
    n[k] = v;
  }
  double best = 180.0;
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j) {
      double d = std::max(-1.0, std::min(1.0, n[i] * n[j]));
      best = std::min(best, 180.0 - std::acos(d) * 180.0 / M_PI);
    }
  return best;
}

Quality quality(const Tr& tr) {
  Quality q;
  for (auto c = tr.finite_cells_begin(); c != tr.finite_cells_end(); ++c) {
    if (!is_domain_cell(tr, c)) continue;
    std::array<Point, 4> p{c->vertex(0)->point(), c->vertex(1)->point(), c->vertex(2)->point(), c->vertex(3)->point()};
    double d = min_dihedral_deg(p);
    q.min_dihedral_deg = std::min(q.min_dihedral_deg, d);
    if (d < 5.0) ++q.below_5;
    if (d < 1.0) ++q.below_1;
    ++q.tets;
    q.volume += std::abs(CGAL::volume(p[0], p[1], p[2], p[3]));
  }
  return q;
}

void write_medit(const Tr& tr, const std::string& path) {
  std::map<Vertex_handle, std::size_t> index;
  std::ofstream os(path);
  os << std::setprecision(17);
  os << "MeshVersionFormatted 2\nDimension 3\n";
  std::size_t nv = 0;
  for (auto v = tr.finite_vertices_begin(); v != tr.finite_vertices_end(); ++v) ++nv;
  os << "Vertices\n" << nv << '\n';
  std::size_t i = 0;
  for (auto v = tr.finite_vertices_begin(); v != tr.finite_vertices_end(); ++v) {
    index[v] = ++i;
    os << v->point().x() << ' ' << v->point().y() << ' ' << v->point().z() << " 0\n";
  }
  std::size_t nc = 0;
  for (auto c = tr.finite_cells_begin(); c != tr.finite_cells_end(); ++c) if (is_domain_cell(tr, c)) ++nc;
  os << "Tetrahedra\n" << nc << '\n';
  for (auto c = tr.finite_cells_begin(); c != tr.finite_cells_end(); ++c) {
    if (!is_domain_cell(tr, c)) continue;
    os << index[c->vertex(0)] << ' ' << index[c->vertex(1)] << ' ' << index[c->vertex(2)] << ' ' << index[c->vertex(3)] << " 1\n";
  }
  os << "End\n";
}

void write_report(const std::string& path, const Quality& before, const Quality& after, std::size_t facets_before,
                  std::size_t facets_after, bool boundary_unchanged, std::size_t verts_before, std::size_t verts_after,
                  const Options& o, double seconds) {
  std::ofstream os(path);
  os << std::setprecision(17) << std::boolalpha << "{\n"
     << "  \"tool\": \"pred777h_sliver_remesh\",\n"
     << "  \"cgal_version\": \"" << CGAL_VERSION_STR << "\",\n"
     << "  \"target_edge_length\": " << o.target_edge_length << ",\n"
     << "  \"iterations\": " << o.iterations << ",\n"
     << "  \"remesh_boundaries\": false,\n"
     << "  \"before\": {\"tets\": " << before.tets << ", \"vertices\": " << verts_before << ", \"min_dihedral_degrees\": " << before.min_dihedral_deg
     << ", \"tets_below_5deg\": " << before.below_5 << ", \"tets_below_1deg\": " << before.below_1 << ", \"volume\": " << before.volume << "},\n"
     << "  \"after\": {\"tets\": " << after.tets << ", \"vertices\": " << verts_after << ", \"min_dihedral_degrees\": " << after.min_dihedral_deg
     << ", \"tets_below_5deg\": " << after.below_5 << ", \"tets_below_1deg\": " << after.below_1 << ", \"volume\": " << after.volume << "},\n"
     << "  \"boundary_facets_before\": " << facets_before << ",\n"
     << "  \"boundary_facets_after\": " << facets_after << ",\n"
     << "  \"boundary_unchanged\": " << boundary_unchanged << ",\n"
     << "  \"seconds\": " << seconds << "\n}\n";
}

// Local validity of the triangulation: what the remeshing actually relies on.
bool local_validity(const Tr& tr, const char* stage) {
  std::size_t bad_orient = 0, bad_neigh = 0, bad_vertex = 0;
  for (auto c = tr.all_cells_begin(); c != tr.all_cells_end(); ++c) {
    for (int i = 0; i < 4; ++i) {
      Cell_handle n = c->neighbor(i);
      if (n == Cell_handle() || !n->has_neighbor(c)) { ++bad_neigh; break; }
      if (c->vertex(i) == Vertex_handle()) { ++bad_neigh; break; }
    }
    if (tr.is_infinite(c)) continue;
    if (CGAL::orientation(c->vertex(0)->point(), c->vertex(1)->point(), c->vertex(2)->point(), c->vertex(3)->point()) != CGAL::POSITIVE) ++bad_orient;
  }
  for (auto v = tr.all_vertices_begin(); v != tr.all_vertices_end(); ++v) {
    Cell_handle c = v->cell();
    if (c == Cell_handle() || !c->has_vertex(v)) ++bad_vertex;
  }
  if (bad_orient || bad_neigh || bad_vertex) {
    std::cerr << stage << ": local validity failed: non-positive finite cells " << bad_orient << ", broken neighbour relations "
              << bad_neigh << ", broken vertex pointers " << bad_vertex << "\n";
    return false;
  }
  return true;
}

void read_medit(const std::string& path, std::vector<Point>& points, std::vector<std::array<int, 5>>& cells) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string tok;
  while (in >> tok) {
    if (tok == "Vertices") {
      std::size_t n; in >> n; points.reserve(n);
      for (std::size_t i = 0; i < n; ++i) { double x, y, z; int ref; in >> x >> y >> z >> ref; points.emplace_back(x, y, z); }
    } else if (tok == "Tetrahedra") {
      std::size_t n; in >> n; cells.reserve(n);
      for (std::size_t i = 0; i < n; ++i) { std::array<int, 5> c; in >> c[0] >> c[1] >> c[2] >> c[3] >> c[4]; for (int k = 0; k < 4; ++k) c[k] -= 1; if (c[4] == 0) c[4] = 1; cells.push_back(c); }
    } else if (tok == "Triangles" || tok == "Edges") {
      std::size_t n; in >> n; std::string line; std::getline(in, line);
      for (std::size_t i = 0; i < n; ++i) std::getline(in, line);
    } else if (tok == "End") break;
  }
  if (points.empty() || cells.empty()) throw std::runtime_error("no vertices or tetrahedra in " + path);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options o = parse(argc, argv);
    const auto t0 = std::chrono::steady_clock::now();
    Tr tr;
    std::size_t reoriented = 0, degenerate = 0;
    {
      std::vector<Point> points; std::vector<std::array<int, 5>> cells;
      read_medit(o.input, points, cells);
      for (auto& c : cells) {
        const CGAL::Orientation ori = CGAL::orientation(points[c[0]], points[c[1]], points[c[2]], points[c[3]]);
        if (ori == CGAL::NEGATIVE) { std::swap(c[1], c[2]); ++reoriented; }
        else if (ori == CGAL::COPLANAR) ++degenerate;
      }
      if (degenerate) throw std::runtime_error("input has " + std::to_string(degenerate) + " exactly degenerate tetrahedra");
      const std::map<std::array<int, 3>, Tr::Cell::Surface_patch_index> border;   // boundary facets are inferred
      std::vector<Vertex_handle> vhv;
      CGAL::build_triangulation<Tr, true>(tr, points, cells, border, vhv, false);
    }
    // Validity.  The global TDS check (Euler relation) is NOT applicable: the material's boundary is a high-genus
    // surface (a Schwarz-P sheet cell has genus 5), and coning it to CGAL's single infinite vertex can never be a
    // 3-sphere.  Everything local is required instead: every finite cell positively oriented, every cell's neighbour
    // relation mutual, every vertex pointing at an incident cell.  The remeshing only touches interior elements, so
    // the singular infinite vertex is never involved in an operation.
    if (!local_validity(tr, "input")) return 4;
    {
      const long V = static_cast<long>(tr.tds().number_of_vertices()), E = static_cast<long>(tr.tds().number_of_edges()),
                 F = static_cast<long>(tr.tds().number_of_facets()), C = static_cast<long>(tr.tds().number_of_cells());
      std::cerr << "triangulation built: reoriented " << reoriented << ", V-E+F-C = " << (V - E + F - C)
                << " (the boundary genus; 0 only for a ball)\n";
    }
    std::size_t verts_before = tr.number_of_vertices();
    const std::set<Facet_key> facets_before = boundary_facets(tr);
    // vertex dimension: 2 on the boundary, 3 inside (the remesher only moves dimension-3 vertices freely)
    for (auto v = tr.finite_vertices_begin(); v != tr.finite_vertices_end(); ++v) v->set_dimension(3);
    for (auto c = tr.finite_cells_begin(); c != tr.finite_cells_end(); ++c) {
      if (!is_domain_cell(tr, c)) continue;
      for (int i = 0; i < 4; ++i) {
        if (is_domain_cell(tr, c->neighbor(i))) continue;
        for (int j = 0; j < 4; ++j) if (j != i) c->vertex(j)->set_dimension(2);
      }
    }
    const Quality before = quality(tr);

    CGAL::tetrahedral_isotropic_remeshing(tr, o.target_edge_length,
        CGAL::parameters::remesh_boundaries(false).number_of_iterations(o.iterations));

    if (!local_validity(tr, "remeshed")) return 4;
    const Quality after = quality(tr);
    const std::set<Facet_key> facets_after = boundary_facets(tr);
    const bool unchanged = (facets_before == facets_after);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!o.report.empty())
      write_report(o.report, before, after, facets_before.size(), facets_after.size(), unchanged, verts_before, tr.number_of_vertices(), o, seconds);
    std::cout << std::setprecision(6)
              << "sliver_remesh: tets " << before.tets << " -> " << after.tets
              << ", min dihedral " << before.min_dihedral_deg << " -> " << after.min_dihedral_deg << " deg"
              << ", below 5 deg " << before.below_5 << " -> " << after.below_5
              << ", below 1 deg " << before.below_1 << " -> " << after.below_1
              << ", volume " << std::setprecision(10) << before.volume << " -> " << after.volume
              << ", boundary " << (unchanged ? "unchanged" : "CHANGED") << ", " << std::setprecision(3) << seconds << " s\n";
    if (!unchanged) { std::cerr << "boundary facets changed: refusing to write the mesh\n"; return 3; }
    write_medit(tr, o.output);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "pred777h_sliver_remesh: " << e.what() << '\n';
    return 2;
  }
}
