This work implements an efficient framework for ranking pre-trained models.

## Environment configuration
install the requirements:
```
pip install -r requirements.txt
```


## Download the models

If you want the trained models in parc, they are available here:
[All Trained Models](https://www.dropbox.com/s/gk32wdqmf19lnmt/models.zip?dl=0). 

## Prepare the data
1. You can check the `dataset.py` to download all the datasets.
2. You can check the `demo.py` to generate all the forward features: `./cache/probes/fixed_budget_500/.....pkl`. \
You can also download the forward features of default 500 samples from here (Recommended): [500_probe_set](https://www.dropbox.com/s/l08n4ejuip2b1h6/probes.zip?dl=0)
```
# note probe_only=True, budget is the size of probe sets.
experiment = Experiment(my_methods, name='test', append=False, budget=500, probe_only=True) 
```
3. You can check the `feature_extractor.py` to generate all clip features.


## Evaluation
See `demo.py` for an example of how to perform evaluation:
```
# All baselines:
python demo.py && python metrics.py

# ours:
python meta_features_plus.py --weight 0.5 --pca_dim 32 --k 5 --alpha 0.0001 --reg 0 --iteration 1000 --seed 2023  --no_completion_rebuilding 'FDA' --proxy_model 'clip' &&  python metric_cold.py 
```


## Add more methods.
You can add your baseline methods in the `methods.py`. We have open-sourced the implementations of baseline methods.

# Fennec benchmark

Fennec benchmark is extended from  [parc](https://arxiv.org/abs/2111.06977) benchmark by including more models and baselines.

## Citation
```
@inproceedings{parc-neurips2021,
  author    = {Daniel Bolya and Rohit Mittapalli and Judy Hoffman},
  title     = {Scalable Diverse Model Selection for Accessible Transfer Learning},
  booktitle = {NeurIPS},
  year      = {2021},
}

@misc{bai2024pretrainedmodelrecommendationdownstream,
      title={Pre-Trained Model Recommendation for Downstream Fine-tuning}, 
      author={Jiameng Bai and Sai Wu and Jie Song and Junbo Zhao and Gang Chen},
      year={2024},
      eprint={2403.06382},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.06382}, 
}
```
