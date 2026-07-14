SRC="$PWD/results/CBSD100_Poisson_blind_3M_restored/test_best/images"
DST="$PWD/CBSD100_Poisson_blind_3M_restored"

mkdir -p "$DST"

for file in "$SRC"/*_recon.png; do
    cp -- "$file" "$DST/$(basename "${file%_recon.png}").png"
done