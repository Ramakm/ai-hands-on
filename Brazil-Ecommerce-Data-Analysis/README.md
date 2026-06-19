# Brazil E-Commerce — Data Analysis

Exploratory data analysis of the **Olist Brazilian E-Commerce** public dataset,
covering customer distribution and geolocation patterns across Brazil. The work
is presented as clean, self-contained Jupyter notebooks with visualizations and
written takeaways.

## Notebooks

| Notebook | What it covers |
| --- | --- |
| [`customer_analysis.ipynb`](customer_analysis.ipynb) | Data quality checks, unique customers vs. orders, customers by state/city, state-level summary, and the share of customers across the top 10 states. |
| [`geolocation_analysis.ipynb`](geolocation_analysis.ipynb) | Coverage by state and city, unique locations per zip-code prefix, a scatter map and 2D density heatmap of Brazil, and a state-level summary. |

## Dataset

This project uses the **Brazilian E-Commerce Public Dataset by Olist**, available on Kaggle:

➡️ https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The raw CSV files are **not** committed to the repository (they total ~120 MB).
To run the notebooks, download the dataset and place the CSVs in a `Data/`
folder next to the notebooks:

```
Brazil-Ecommerce-Data-Analysis/
├── customer_analysis.ipynb
├── geolocation_analysis.ipynb
└── Data/
    ├── olist_customers_dataset.csv
    ├── olist_geolocation_dataset.csv
    ├── olist_orders_dataset.csv
    ├── olist_order_items_dataset.csv
    ├── olist_order_payments_dataset.csv
    ├── olist_order_reviews_dataset.csv
    ├── olist_products_dataset.csv
    ├── olist_sellers_dataset.csv
    └── product_category_name_translation.csv
```

## Setup

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook
```

Then open either notebook and run all cells.

## Key takeaways

- Customers and orders are heavily concentrated in **São Paulo (SP)**, followed by
  Rio de Janeiro and Minas Gerais — a small set of states accounts for most activity.
- Geolocation coverage mirrors the customer distribution, with dense clustering
  along Brazil's southeastern coast visible in both the scatter map and density heatmap.

See each notebook's final "Key takeaways" section for the full discussion.
