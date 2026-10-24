# Random K-fold vs Time-aware Validation: L&T Vehicle Loan Default

The same loan-default model is built twice, from the same data and with the same rules. The only difference is how each version checks itself during development:

- **Random K-fold:** shuffle all loans and test on random slices. This is the standard approach.
- **Time-aware:** train on earlier loans and test on the weeks that follow, as happens after deployment.

Both final models are then tested on loans from the last nine days of October 2018, a period neither has seen.

The model predicts whether a vehicle loan defaults on its **first EMI**, using what is known at disbursal: loan size, loan-to-value, credit-bureau history, borrower details, dealer and branch. The data is Kaggle's *L&T Vehicle Loan Default Prediction* set: 233,154 labelled loans disbursed between 1 August and 31 October 2018.

Performance is measured by **AUC**: how well the model ranks risky loans above safe ones (0.5 is a coin toss, 1.0 is perfect).

---

## Summary

The two approaches generate near similar results but, the time-aware approach is more honest and reports patterns random K-fold cannot see.

1. **Same final model, near-identical real-world performance.** Both approaches ended up with CatBoost on all 48 features, scoring AUC 0.660 (random K-fold) and 0.662 (time-aware) on the test loans.
2. **Time-aware validation found patterns random K-fold missed:**
   - features whose value fades over time;
   - a shift away from *where* a loan came from towards the *borrower's credit behaviour*;
   - a simpler and cheaper model with no loss in accuracy.

| | Random K-fold | Time-aware |
|---|---|---|
| Final model | CatBoost, all 48 features | CatBoost, all 48 features |
| Forecast AUC (from validation) | 0.679 | 0.670 |
| **Actual AUC on 23–31 Oct** | **0.660** | **0.662** |
| How much the forecast overstated | 0.018 | 0.009 |
| Features flagged as fading over time | 0 | 3 |
| Features in the LightGBM model | 48 | 15 |

---

## 1. The final model came out the same

Both approaches chose **CatBoost using all 48 candidate features**.
- They scored **AUC 0.6605** (random K-fold) and **0.6620** (time-aware) on the October test loans.
- The difference (time-aware minus random K-fold) was +0.0011 (95% interval +0.0002 to +0.0022).
- Their predictions correlate at 0.991.

## 2. Inferences from Time-aware validation

| | Forecast | Actual | Overstated by |
|---|---|---|---|
| Random K-fold (CatBoost) | 0.679 | 0.660 | 0.018 |
| Time-aware (CatBoost) | 0.670 | 0.662 | 0.009 |
| Random K-fold (simple LightGBM benchmark) | 0.661 | 0.647 | 0.014 |
| Time-aware (simple LightGBM benchmark) | 0.656 | 0.647 | 0.009 |

Random K-fold promised more than the model delivered. Time-aware validation cut that overstatement roughly in half for the chosen model, and by about a third for a simple benchmark model.

Even the time-aware forecast was a little high. The model's accuracy slipped steadily the further ahead it had to predict: AUC 0.682 for early September, 0.663 by mid-October. An average over past periods therefore runs slightly above what the next period delivers.

## 3. Feature selection

**The key feature:** Loan-to-value (`ltv`) was the strongest feature everywhere. After the filters, both kept 24 features, 22 of them the same.

**Time-aware validation spots degrading features.** It flagged three features whose predictive value faded in later weeks, including the borrower's pincode. Random K-fold flagged none. It can't see this kind of decay, because it tests on loans from the same weeks it trained on.

**Random K-fold leans on *where* the loan came from; time-aware leans on the *borrower's credit behaviour*.** In the CatBoost rankings:

| Feature | Random K-fold rank | Time-aware rank |
|---|---|---|
| Branch (`branch_id`) | 5 | 19 |
| Dealer (`supplier_id`) | 10 | 20 |
| Pincode (`Current_pincode_ID`) | 6 | 14 |
| Share of overdue accounts (`pri_overdue_share`) | 12 | 4 |
| Disbursed vs sanctioned on past loans (`pri_disb_to_sanc`) | 11 | 5 |
| Recent credit enquiries (`NO.OF_INQUIRIES`) | 16 | 9 |

This fits a simple story. Branch, dealer and location patterns are partly tied to particular weeks, so they look more useful when the test loans come from the same weeks as the training loans. A borrower's credit history carries over to new periods better. (The loan officer, `Employee_code_ID`, ranked 2nd under both approaches; it is the exception.)

**Time-aware validation allowed a simpler model.** For LightGBM, it settled on **15 features**; random K-fold insisted on all 48. On the test loans, the 15-feature model was **just as good**: AUC 0.6471 vs 0.6470, a difference of +0.0001 (95% interval ±0.003).

Random K-fold kept everything because its five test slices are near-copies of each other. Tiny differences look decisive, so the selection rule never drops a feature. (CatBoost still kept all 48 features under both approaches.)

## 4. Model choice

CatBoost beat LightGBM under both approaches, by a wide and consistent margin:

| | Validation margin | Folds won | Margin on test |
|---|---|---|---|
| Random K-fold | +0.018 | 5 of 5 | +0.013 |
| Time-aware | +0.015 | 4 of 4 | +0.014 |

On test, both margins are clearly above zero. When one model type is clearly stronger, the validation approach doesn't change the decision.

---

## Conclusion

Even though random K-fold and time-aware methodologies scored nearly same, the time-aware modelling is more accurate because:
1. It unearths feature degradation that can be critical for longer times of data (This dataset only contains 3 months of data).
2. Its testing performance is closer to its validation performance as compared to random k-fold, depicting a better fitted model.
3. Cost of training is lesser as the time and resources consumed to train time-aware model was less, making it more feasible for incoming data.  

---

## How to run

| | Random K-fold | Time-aware |
|---|---|---|
| Validation | 5 stratified random folds | 4 forward-chaining blocks (Sep–Oct): train on the past, validate on the next block |
| Early-stopping rows | A random 10% of the remaining rows | The 5 days just before the validation block |

**Everything else is identical:** data, the 48 candidate features, encodings, model settings, filter thresholds, the feature-selection rule, the model-choice rule and the final-model procedure.

Each approach runs the full workflow on its own:
1. **Feature selection:** Information Value filters, then model-based selection.
2. **Model choice:** LightGBM vs CatBoost.
3. **Final model.**
4. **One evaluation on the test window.**

| | Period | Loans |
|---|---|---|
| Train | 1 Aug – 22 Oct 2018 | 182,501 |
| Test | 23 – 31 Oct 2018 | 50,653 |

Test labels are kept in a separate file and read only by the final notebook, where the test window is scored once. Both approaches start from byte-identical data files: the SHA-256 hashes recorded in the two `manifest.json` files match.

## Project structure

```
Data/
  Raw/                               Kaggle files (train.csv is used; test.csv has no labels)
  Processed_kfold_cv/                features, labels, selection, CV results, models, test results
  Processed_time_cv/                 the same, for the time-aware approach
Notebooks_kfold_cv/                  random K-fold
  00_data_prep.ipynb                 date split, feature engineering, drift diagnostics
  01_feature_selection.ipynb         filters and model-based selection on this approach's folds
  02_model_choice_and_final.ipynb    LightGBM vs CatBoost, final models
  03_test_evaluation.ipynb           scores the test window once
Notebooks_time_cv/                   time-aware: the same four notebooks
requirements.txt
```

## How to run

1. Download the Kaggle dataset `mamtadhaker/lt-vehicle-loan-default-prediction` into `Data/Raw/`. `kagglehub`, listed in `requirements.txt`, can do this.
2. `pip install -r requirements.txt`
3. In each notebook folder, run `00` → `01` → `02` → `03` in order.

note: The code is written to run CatBoost explicitly on a GPU. Please modify this as per your needs