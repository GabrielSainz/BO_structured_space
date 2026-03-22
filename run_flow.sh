#!/usr/bin/env bash
for i in {1..5}
do
    python run.py --function-name albuterol_similarity --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash

    python run.py --function-name amlodipine_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
        
    python run.py --function-name celecoxib_rediscovery --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
        
    python run.py --function-name deco_hop --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
        
    #python run.py --function-name drd2_docking --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name fexofenadine_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name gsk3_beta --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name isomer_c7h8n2o2 --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name isomer_c9h10n2o2pf2cl --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name jnk3 --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name median_1 --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name median_2 --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name mestranol_similarity --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name osimetrinib_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name perindopril_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name ranolazine_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name rdkit_logp --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name rdkit_qed --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name sa_tdc --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name scaffold_hop --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name sitagliptin_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name thiothixene_rediscovery --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name troglitazone_rediscovery --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name valsartan_smarts --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
    #    
    #python run.py --function-name zaleplon_mpo --solver-name cowboys_flow --max-iter 300 --seed $i --no-strict-on-hash
done