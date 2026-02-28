# Data Preprocessing Pipeline

## Dataset

Titanic dataset used to demonstrate preprocessing techniques including cleaning, encoding and scaling.

---

## Conclusion

### Missing Values

Median was better for numerical features because outliers were present and mean would be affected.
Mode preserved categorical distributions.

### Categorical Encoding

* One-Hot Encoding worked best for nominal features (no order)
* Ordinal Encoding suited ordered categories
* Label Encoding simple binary categories
* Frequency Encoding reduced dimensionality
* Target Encoding effective for high cardinality columns

### Feature Scaling

Standardization performed best after log transformation since features followed near-normal distribution.

### Outliers

IQR method removed extreme values improving data consistency.

### Skewness

Log transformation reduced skewness and stabilized variance of numerical features.

---

This project demonstrates a complete preprocessing pipeline that prepares raw data for machine learning models.
