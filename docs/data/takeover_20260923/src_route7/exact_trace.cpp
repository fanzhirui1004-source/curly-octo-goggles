// Route 7: exact rational elimination of the complete-trace functionals, same rules as the frozen
// stage_cutfem_multiconstraint.pivot_coordinates (max-pivot v2 and the historical first-pivot replay) and
// stage_cutfem_full_interface / pivot_coordinates.direct_sum_with_pivots, in C++ with GMP rationals.
//
// Exact arithmetic makes every output unique given the pivot sequence: a reduced row is the unique vector in
// row + span(earlier basis rows) with zeros at all earlier pivots, and the pivot rule (largest |value|, ties to the
// smallest column; or the smallest column) is applied to that unique vector. So these outputs must equal the
// frozen Python ones bit for bit; the caller checks that against the frozen implementation.
//
// Modes: max (max pivot, with the combination rows of the audit record), maxnc (max pivot without combinations: the
// coordinates P, inverse and L depend only on the basis rows and pivots), first (historical first-pivot replay).
// Input (text, stdin):  mode(max|maxnc|first) n_columns n_rows, then per row: k (col value)*k, value as GMP "num/den".
// Output (text, stdout): selected rows, pivots, basis rows, combination rows; for mode max also the direct sum
// (q and null maps). Values are printed in canonical GMP form, which equals str() of gmpy2.mpq / Fraction.
#include <gmp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <queue>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <functional>

struct Entry { int col; mpq_t v; };
typedef std::vector<Entry> Row;  // sorted by col

static void row_free(Row &r) { for (auto &e : r) mpq_clear(e.v); r.clear(); }

// Dense working accumulator with a nonzero list.
struct Work {
    std::vector<mpq_t> val; std::vector<char> on; std::vector<int> cols; int n;
    explicit Work(int n_) : val(n_), on(n_, 0), n(n_) { for (int i = 0; i < n; ++i) mpq_init(val[i]); }
    ~Work() { for (int i = 0; i < n; ++i) mpq_clear(val[i]); }
    void load(const Row &r) { for (auto &e : r) { mpq_set(val[e.col], e.v); if (!on[e.col]) { on[e.col] = 1; cols.push_back(e.col); } } }
    bool has(int c) const { return on[c] && mpq_sgn(val[c]) != 0; }
    // this -= f * r
    void axpy(const mpq_t f, const Row &r, mpq_t tmp) {
        for (auto &e : r) {
            mpq_mul(tmp, f, e.v);
            if (!on[e.col]) { on[e.col] = 1; cols.push_back(e.col); mpq_set_ui(val[e.col], 0, 1); }
            mpq_sub(val[e.col], val[e.col], tmp);
        }
    }
    // gather nonzeros (sorted by col) and reset
    Row take() {
        std::sort(cols.begin(), cols.end());
        Row out;
        for (int c : cols) {
            if (mpq_sgn(val[c]) != 0) { Entry e; e.col = c; mpq_init(e.v); mpq_set(e.v, val[c]); out.push_back(e); }
            mpq_set_ui(val[c], 0, 1); on[c] = 0;
        }
        cols.clear();
        return out;
    }
};

static Row read_row(FILE *in, std::vector<char> &buf) {
    int k; if (fscanf(in, "%d", &k) != 1) { fprintf(stderr, "bad row\n"); exit(2); }
    Row r(k);
    for (int i = 0; i < k; ++i) {
        if (fscanf(in, "%d %s", &r[i].col, buf.data()) != 2) { fprintf(stderr, "bad entry\n"); exit(2); }
        mpq_init(r[i].v);
        if (mpq_set_str(r[i].v, buf.data(), 10) != 0) { fprintf(stderr, "bad rational %s\n", buf.data()); exit(2); }
        mpq_canonicalize(r[i].v);
    }
    std::sort(r.begin(), r.end(), [](const Entry &a, const Entry &b) { return a.col < b.col; });
    return r;
}

static void print_row(FILE *out, const Row &r) {
    fprintf(out, "%zu", r.size());
    for (auto &e : r) { fprintf(out, " %d ", e.col); mpq_out_str(out, 10, e.v); }
    fputc('\n', out);
}

int main(int argc, char **argv) {
    std::vector<char> buf(1 << 20);
    char mode[16]; int ncol, nrow;
    if (scanf("%15s %d %d", mode, &ncol, &nrow) != 3) { fprintf(stderr, "bad header\n"); return 2; }
    bool maxpivot = std::strcmp(mode, "max") == 0 || std::strcmp(mode, "maxnc") == 0;
    bool track = std::strcmp(mode, "max") == 0;
    std::vector<Row> rows(nrow);
    for (int k = 0; k < nrow; ++k) rows[k] = read_row(stdin, buf);

    std::vector<Row> basis, comb;
    std::vector<int> pivots, selected;
    std::vector<int> position(ncol, -1);  // pivot column -> basis index
    Work w(ncol), wc(nrow);
    std::vector<char> scheduled;
    mpq_t tmp, factor, absbest, absv, scale; mpq_init(tmp); mpq_init(factor); mpq_init(absbest); mpq_init(absv); mpq_init(scale);
    for (int k = 0; k < nrow; ++k) {
        w.load(rows[k]);
        Row unit(1); unit[0].col = k; mpq_init(unit[0].v); mpq_set_ui(unit[0].v, 1, 1);
        if (track) wc.load(unit);
        row_free(unit);
        // heap over basis indices of pivots present in the row (same visiting order as the frozen code)
        std::priority_queue<int, std::vector<int>, std::greater<int>> pending;
        std::vector<int> sched_list;
        if (scheduled.size() < basis.size()) scheduled.resize(basis.size() + 1024, 0);
        for (int c : w.cols) { int p = position[c]; if (p >= 0 && !scheduled[p]) { scheduled[p] = 1; sched_list.push_back(p); pending.push(p); } }
        while (!pending.empty()) {
            int idx = pending.top(); pending.pop();
            int piv = pivots[idx];
            if (!w.has(piv)) continue;
            mpq_set(factor, w.val[piv]);
            w.axpy(factor, basis[idx], tmp);
            if (track) wc.axpy(factor, comb[idx], tmp);
            for (auto &e : basis[idx]) {
                int nx = position[e.col];
                if (nx > idx && !scheduled[nx] && w.has(e.col)) { scheduled[nx] = 1; sched_list.push_back(nx); pending.push(nx); }
            }
        }
        for (int p : sched_list) scheduled[p] = 0;
        Row r = w.take();
        Row cr; if (track) cr = wc.take();
        if (r.empty()) { row_free(cr); continue; }
        int best = -1;
        if (maxpivot) {
            for (auto &e : r) {
                mpq_abs(absv, e.v);
                if (best < 0 || mpq_cmp(absv, absbest) > 0) { best = e.col; mpq_set(absbest, absv); }  // ties keep the smaller col
            }
        } else {
            best = r.front().col;
        }
        for (auto &e : r) if (e.col == best) { mpq_set(scale, e.v); break; }
        for (auto &e : r) mpq_div(e.v, e.v, scale);
        for (auto &e : cr) mpq_div(e.v, e.v, scale);
        position[best] = (int)pivots.size();
        pivots.push_back(best); basis.push_back(std::move(r)); comb.push_back(std::move(cr)); selected.push_back(k);
    }
    printf("selected %zu\n", selected.size());
    for (size_t i = 0; i < selected.size(); ++i) printf("%d %d\n", selected[i], pivots[i]);
    if (!maxpivot) return 0;
    printf("basis\n"); for (auto &r : basis) print_row(stdout, r);
    if (track) { printf("combinations\n"); for (auto &r : comb) print_row(stdout, r); }

    // direct_sum_with_pivots: augmented rows [basis_i, e_(ncol+i)], reverse elimination of later pivots
    int nb = (int)basis.size();
    std::vector<Row> aug(nb);
    for (int i = 0; i < nb; ++i) {
        for (auto &e : basis[i]) { Entry x; x.col = e.col; mpq_init(x.v); mpq_set(x.v, e.v); aug[i].push_back(x); }
        Entry x; x.col = ncol + i; mpq_init(x.v); mpq_set_ui(x.v, 1, 1); aug[i].push_back(x);
    }
    std::vector<std::vector<int>> dependents(nb);
    for (int i = 0; i < nb; ++i)
        for (auto &e : basis[i]) { int t = position[e.col]; if (t > i) dependents[t].push_back(i); }
    Work wa(ncol + nb);
    for (int i = nb - 1; i >= 0; --i) {
        int piv = pivots[i];
        for (int j : dependents[i]) {
            // if pivot in augmented[j]: augmented[j] -= augmented[j][pivot] * augmented[i]
            Entry *hit = nullptr;
            for (auto &e : aug[j]) if (e.col == piv) { hit = &e; break; }
            if (!hit || mpq_sgn(hit->v) == 0) continue;
            mpq_set(factor, hit->v);
            wa.load(aug[j]);
            wa.axpy(factor, aug[i], tmp);
            row_free(aug[j]);
            aug[j] = wa.take();
        }
    }
    printf("augmented\n"); for (auto &r : aug) print_row(stdout, r);
    return 0;
}
