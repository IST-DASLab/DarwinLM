1. Run ziplm.sh script to produce database (so far you need to manually specify the number of columns removed for each compression step; timing not implemented yet)
2. Run struct_prune_search.sh script for the search 
3. Run run_lmeval for evaluation (evo_struct_prune_search.py produces a configuration file that you need to pass as sparse_config_path)
