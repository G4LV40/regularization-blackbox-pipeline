# 📘 Regularized and Hybrid Modeling with Machine Learning

This repository contains the code used in the paper  
**_"Non-Heuristic Variable Selection via Regularization for Black Box Models in Travel Insurance Contracting Prediction"_**,  

---

## 📁 Main Scripts

| File Name             | Brief Description |
|-----------------------|-------------------|
| `create_dt_set.R`     | Generates datasets from original and engineered sources. |
| `Elastic_Net_Class.R` | Trains and validates the Elastic Net classification model. |
| `funct_Models_.R`     | Auxiliary functions for fitting and evaluating various models. |
| `funct_RND_search.R`  | Implements random hyperparameter search with cross-validation. |
| `hybrid_models.R`     | Integrates regularized models with non-linear algorithms (hybrid models). |
| `Lasso_class.R`       | Trains and evaluates LASSO models for classification tasks. |
| `modelos_bb_class.R`  | Trains black-box models (XGBoost, CatBoost, etc.). |
| `Ridge_Class.R`       | Applies Ridge regularization for classification problems. |
| `simulation_.R`       | Runs simulations (varying n and p) for empirical method evaluation. |

---

## 📚 Citation

If you use this code in your research, please cite it as:

```bibtex
@misc{galvao&moral2025,
  author       = {Luciano Galvão & Rafael Moral},
  title        = {Hybrid Non-Heuristic Variable Selection Models via Regularization for
Black Box Models Applied to the Insurance Sector},
  year         = {2025},
  howpublished = {\url{https://github.com/seuusuario/artigo1-variaveis-blackbox}},
  note         = {Code repository associated with Article 1 of the Ph.D. thesis.}
}
