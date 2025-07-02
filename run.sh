#!bash

python3 main.py --model pipeline --normalization RLAE --k 10 20 &&
python3 main.py --model combine --normalization EASE --k 10 20 &&
python3 main.py --model regularization --normalization LAE --k 10 20 &&
python3 main.py --model dan --normalization RLAE --k 10 20 &&
python3 main.py --model sapling --k 10 20 &&
python3 main.py --model dan --normalization EASE --k 10 20