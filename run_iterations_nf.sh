#!/bin/bash
for i in {1..10}
do
    python run.py --function-name osimetrinib_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow_iteration" --save-iteration-results --checkpoint-every 1

    python run.py --function-name median_2 --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow_iteration" --save-iteration-results --checkpoint-every 1

    python run.py --function-name amlodipine_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow_iteration" --save-iteration-results --checkpoint-every 1

    python run.py --function-name perindopril_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow_iteration" --save-iteration-results --checkpoint-every 1

    python run.py --function-name ranolazine_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow_iteration" --save-iteration-results --checkpoint-every 1
        
    python run.py --function-name zaleplon_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash --sufix "nflow_iteration" --save-iteration-results --checkpoint-every 1
done
