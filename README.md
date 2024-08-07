All Python code was tested with Python 3.6.10. The scripts were run on a Mac-OS operating system.

To reproduce our Python environment, you can use the .txt file we provide by running:
```
pip install -r requirements.txt
```


In the root folder, we provide 4 main Python scripts:

- Regression (Simulation): In **regression.py**, we use a synthetic regression example to illustrate the performance of different evaluation procedures. We can vary ``model``, ``sample_size``, ``fold_num``, ``nested_cv``, ``feature_num`` to incorporate different scenarios. Here, the ``model`` config is chosen from the list ``['rf', 'knn', 'ridge']``. And ``nested_cv = True`` means including  evaluating using the nested_cv procedure in [B2023] while excluding the LOOCV procedure; and set ``nested_cv = False`` means including the LOOCV procedure while excluding the nested_cv procedure.

- Real-world Regression: In **real_world_regression.py**, we evaluate the `knn` model performance within the dataset [puma32H](https://wwww.openml.org/d/1210). We can vary ``model``, ``sample_size``, ``fold_num``, ``nested_cv``, ``feature_num`` to control effects of different variables. To replicate our result, you need to download the ``BNG_puma32H.arff`` from the website and store it in ``src/datasets``. At each random seed, the whole dataset is shuffled, and we choose the first rows (``sample_num``) to train each model and use the remaining dataset to approximate the true model performance.

- Portfolio: In **portfolio.py**, we do the CVaR-portfolio experiment. The standard case includes plug-in, $K$-fold CV and LOOCV by specifying different $K$ = fold_num. We can vary ``model``, ``sample_size``, ``fold_num`` to incorporate different scenarios. The ``model`` config is chosen from the list ``['SAA', 'kNN']``.

- Newsvendor (Appendix G.3): In **newsvendor.py**: we use ``base_nv.py`` and ``shift_alg_nv.py`` to denote the base and shift case respectively. The standard case includes plug-in, $K$-fold CV and LOOCV by specifying different $K$ = fold_num.  We can vary ``model``, ``sample_size``, ``fold_num`` to incorporate different scenarios. The ``model`` config is chosen from the list ``['kNN', 'LERM', 'rf']``, where ``rf`` denotes the implemented ``Stochastic Optimization Forest`` using the codebase: https://github.com/CausalML/StochOptForest. And ``compare_alg_nv.py`` is used to conduct hypothesis testing for Figure 3 in Appendix G.3.



**Note**: 

With the files provided, you can recover our results following the instructions below provided. Due to the existence of random seeds, the results may not be exactly the same.

Start a new Terminal and run the code below:

```
python regression.py --model rf --sample_size 10000 --fold_num 10
```

In the ``src`` folder, we include: 
- The folder ``datasets``: help load the original file of the real-world datasets;
- The folder ``StochOptForest``: the implemented ``Stochastic Optimization Forest`` using the codebase:
https://github.com/CausalML/StochOptForest.
- Basic model implementattions in each task: Regression (``regression_model.py``); Portfolio (``model_portfolio.py``); Newsvendor（``model_nv.py``).
- Some complex interval constructions: ``CI_construction.py`` for covariate shifts; ``nested_cv.py`` for nested cv, where we set the random split number ``split_num = 20`` instead of 1000 suggested by [B2023] to reduce computational costs.
- And some other utlity files.

Reference:

[B2023] Bates, Stephen, Trevor Hastie, and Robert Tibshirani. "Cross-validation: what does it estimate and how well does it do it?." Journal of the American Statistical Association 119.546 (2024): 1434-1445.
