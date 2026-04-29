#!/bin/bash
# chmod +x run_iterations_nf.sh
# ./run_iterations_nf.sh
for i in {1..5}
do
    python run.py --function-name osimetrinib_mpo --solver-name cowboys_flow_2 --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow3_iteration2" --save-iteration-results --checkpoint-every 1

    python run.py --function-name median_2 --solver-name cowboys_flow_2 --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow3_iteration2" --save-iteration-results --checkpoint-every 1

    python run.py --function-name amlodipine_mpo --solver-name cowboys_flow_2 --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow3_iteration2" --save-iteration-results --checkpoint-every 1

    python run.py --function-name perindopril_mpo --solver-name cowboys_flow_2 --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow3_iteration2" --save-iteration-results --checkpoint-every 1

    python run.py --function-name ranolazine_mpo --solver-name cowboys_flow_2 --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow3_iteration2" --save-iteration-results --checkpoint-every 1
        
    python run.py --function-name zaleplon_mpo --solver-name cowboys_flow_2 --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow3_iteration2" --save-iteration-results --checkpoint-every 1
done
