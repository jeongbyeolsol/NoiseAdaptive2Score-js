SRC="$PWD/results/CBSD100_07M_smooth_009_051_b16/test_best/images"
DST="$PWD/CBSD100_07M_smooth_009_051"

mkdir -p "$DST"

for file in "$SRC"/*_recon.png; do
    cp -- "$file" "$DST/$(basename "${file%_recon.png}").png"
done