# Tips
   [![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)
   
Tips is a smart GNN recommender system library which helps choose recommender systems with most optimised results. 

Tips is a use-case centric library and the end-goal is to implement GNN based recommender systems with just 2-3 lines 
of code, compare them and protoype the best working model.
```python
from tips import Tips, TipsFactory

properties_dict = {"model": "LightGCN", "data_path":"/path/to/data"}
new_tip = TipsFactory(properties_dict)
train, valid, test = Tips.create_dataset(new_tip)
model, best_score, valid_score = Tips.train(Tips.get_model(new_tip))
Tips.eval_model(model)
```

An easier approach is to build directly from the properties dict.
```python
from tips import Tips

properties_dict = {"model": "LightGCN", "data_path":"/path/to/data"}
model, best_score, valid_score = Tips.train_model(properties_dict)
Tips.eval_model(model)
```
This library is WIP.