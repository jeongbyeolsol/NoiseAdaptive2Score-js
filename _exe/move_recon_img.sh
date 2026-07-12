SRC="$PWD/results/CBSD100_Poisson_restored/test_best/images"
DST="$PWD/results/CBSD100_Poisson_restored/restored_only"

mkdir -p "$DST"

for file in "$SRC"/*_recon.png; do
    cp -- "$file" "$DST/$(basename "${file%_recon.png}").png"
done