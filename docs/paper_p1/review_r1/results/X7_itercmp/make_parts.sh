#!/bin/bash
# make_parts.sh <lattice> : on the bjb2 host, split the per-cell systems and small files of sys_<lattice> into 256 MB parts
# under parts_<lattice>/ with a manifest "<part> <md5> <target file>" (the whole-lattice K is rebuilt from the cells).
set -e
L=$1; S=/root/autodl-tmp/ITERCMP/sys_$L; P=/root/autodl-tmp/ITERCMP/parts_$L
rm -rf $P; mkdir -p $P; cd $S
: > $P/manifest.txt
for f in cell*.npz b.npy F.npy xyz.npy comp.npy meta.json; do
  split -b 64M -d -a 3 $f $P/$f.part
  for p in $P/$f.part*; do echo "$(basename $p) $(md5sum < $p | cut -d' ' -f1) $f" >> $P/manifest.txt; done
done
md5sum cell*.npz b.npy F.npy xyz.npy comp.npy meta.json > $P/whole.md5
echo "parts=$(wc -l < $P/manifest.txt) bytes=$(du -sb $P | cut -f1)"
