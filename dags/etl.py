from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import task
from airflow.sdk import task_group
from airflow.sdk import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import logging
from sqlalchemy import create_engine

coingecko_api_key = Variable.get("COINGECKO_API_KEY")
db_user = Variable.get("DB_USER")
db_password = Variable.get("DB_PASSWORD")
db_host = Variable.get("DB_HOST")
db_name = Variable.get("DB_NAME")
db_port = Variable.get("DB_PORT", default="5432")
conn_id = "postgres_conn"

cfg = {
    "dag_id": "coingecko_shopping_dag",
    "start_date": datetime(2025, 1, 1),
    "schedule": "@daily",
    "catchup": False,
    "default_args": {
        "owner": "airflow",
        "depends_on_past": False,
        "retries": 0,
    },
}

shopping_behavior_attributes = [
    "Customer ID",
    "Age",
    "Gender",
    "Item Purchased",
    "Category",
    "Purchase Amount (USD)",
    "Location",
    "Size",
    "Color",
    "Season",
    "Review Rating",
    "Subscription Status",
    "Shipping Type",
    "Discount Applied",
    "Promo Code Used",
    "Previous Purchases",
    "Payment Method",
    "Frequency of Purchases",
]

shopping_behavior_numerical_attributes = [
    "Customer ID",
    "Age",
    "Purchase Amount (USD)",
    "Review Rating",
    "Previous Purchases",
]

shopping_behavior_categorical_attributes = [
    "Gender",
    "Item Purchased",
    "Category",
    "Location",
    "Size",
    "Color",
    "Season",
    "Subscription Status",
    "Shipping Type",
    "Discount Applied",
    "Promo Code Used",
    "Payment Method",
    "Frequency of Purchases",
]

# Help functions
def fetch_data(route):
    url = f"https://api.coingecko.com/api/v3/{route}?key={coingecko_api_key}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def create_social_generations(age):
    if age <= 28:
        return 'Gen Z'
    elif age <= 44:
        return 'Millennial'
    elif age <= 60:
        return 'Gen X'
    elif age <= 79:
        return 'Baby Boomer'
    else:
        return 'Silent Generation'

def purchase_amount_category(amount):
    if amount <= 30:
        return 'Low ($0-30)'
    elif amount <= 50:
        return 'Medium ($31-50)'
    elif amount <= 80:
        return 'High ($51-80)'
    else:
        return 'Very High ($81+)'

def state_color(state):
    blue_states = [
        'California', 'Colorado', 'Connecticut', 'Delaware', 'Hawaii', 'Illinois',
        'Maine', 'Maryland', 'Massachusetts', 'Minnesota', 'New Hampshire',
        'New Jersey', 'New Mexico', 'New York', 'Oregon', 'Rhode Island',
        'Vermont', 'Virginia', 'Washington'
    ]

    red_states =    [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'Florida', 'Georgia',
        'Idaho', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
        'Michigan', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
        'Nevada', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
        'Pennsylvania', 'South Carolina', 'South Dakota', 'Tennessee',
        'Texas', 'Utah', 'West Virginia', 'Wisconsin', 'Wyoming'
    ]

    if state in blue_states:
        return 'Blue'
    elif state in red_states:
        return 'Red'
    else:
        logging.warning(f"State color not found for: {state}")
        return 'Unknown'

# @task
# def fetch_data_shopping():
#     file_path = "shopping_behavior_updated.csv"
#     df = kagglehub.dataset_load(
#         adapter=KaggleDatasetAdapter.PANDAS,
#         handle="ahmadrazakashif/shopping-behavior-dataset",
#         path=file_path,
#     )
#     print("Shopping data read:")
#     print(df.head())
#     tbl_dict = df.to_dict('dict')
#     return tbl_dict

# Task definitions
@task
def fetch_data_shopping():
    file_path = "data/shopping_behavior_updated.csv"
    df = pd.read_csv(file_path)

    logging.info("Shopping data read:")
    logging.info(df.head())

    tbl_dict = df.to_dict('dict')
    return tbl_dict

@task
def fetch_data_coins():
    global_data = fetch_data("global")
    trump_data = fetch_data("coins/categories/trump-affiliated-tokens")
    rwa_data = fetch_data("coins/categories/real-world-assets-rwa")
    solana_meme_data = fetch_data("coins/categories/solana-meme-coins")

    data = {
        "global_data": global_data,
        "trump_data": trump_data,
        "rwa_data": rwa_data,
        "solana_meme_data": solana_meme_data,
    }

    logging.info(f"Fetched {len(data['global_data'])} global data")
    logging.info(f"Fetched {len(data['trump_data'])} trump data")
    logging.info(f"Fetched {len(data['rwa_data'])} rwa data")
    logging.info(f"Fetched {len(data['solana_meme_data'])} solana meme data")

    return data

@task
def rename_columns(tbl_dict):
    df = pd.DataFrame(tbl_dict)

    df = df.rename(columns={
        "Purchase Amount (USD)": "price",
        "Customer ID": "customer_id",
        "Item Purchased": "item_purchased",
        "Review Rating": "rating",
        "Subscription Status": "subscription_status",
        "Shipping Type": "shipping_type",
        "Previous Purchases": "previous_purchases",
        "Discount Applied": "discount_applied",
        "Promo Code Used": "promo_code_used",
        "Payment Method": "payment_method",
        "Frequency of Purchases": "purchase_frequency",
        "Age": "age",
        "Location": "location",
        "Gender": "gender",
        "Category": "category",
        "Size": "size",
        "Color": "color",
        "Season": "season",
    })

    tbl_dict_renamed = df.to_dict('dict')
    return tbl_dict_renamed

@task
def compute_extra_columns(tbl_dict):
    df = pd.DataFrame.from_dict(tbl_dict)

    df['social_generation'] = df['age'].apply(create_social_generations)
    df['social_generation'] = df['social_generation'].astype('category')

    df['amount_category'] = df['price'].apply(purchase_amount_category)
    df['amount_category'] = df['amount_category'].astype('category')

    df['state_color'] = df['location'].apply(state_color)
    df['state_color'] = df['state_color'].astype('category')

    tbl_dict_updated = df.to_dict('dict')
    return tbl_dict_updated

@task
def clean_coin_data(data):

    logging.info("Coin data: {}".format(data))

    global_data = data["global_data"]
    trump_data = data["trump_data"]
    rwa_data = data["rwa_data"]
    solana_meme_data = data["solana_meme_data"]

    timestamp = global_data["data"]["updated_at"]
    global_market_cap_24h_change = global_data["data"]["market_cap_change_percentage_24h_usd"]
    global_market_cap_24h_change = round(global_market_cap_24h_change, 2)
    trump_market_cap_24h_change = trump_data["market_cap_change_24h"]
    trump_market_cap_24h_change = round(trump_market_cap_24h_change, 2)
    rwa_market_cap_24h_change = rwa_data["market_cap_change_24h"]
    rwa_market_cap_24h_change = round(rwa_market_cap_24h_change, 2)
    solana_meme_market_cap_24h_change = solana_meme_data["market_cap_change_24h"]
    solana_meme_market_cap_24h_change = round(solana_meme_market_cap_24h_change, 2)

    clean_data = {
        "timestamp": timestamp,
        "global_change": global_market_cap_24h_change,
        "trump_change": trump_market_cap_24h_change,
        "rwa_change": rwa_market_cap_24h_change,
        "meme_change": solana_meme_market_cap_24h_change,
    }

    return clean_data

# Looking for a correlation between meme coin performance and spending habits for different social generations
# 24h meme coins change% * each generation spending% (compared to the average)
@task
def plot_generation_meme_coins_correlation(tbl_dict, coin_data):
    df = pd.DataFrame.from_dict(tbl_dict)
    timestamp = coin_data["timestamp"]

    os.makedirs('/images/', exist_ok=True)
    os.makedirs(f'/images/{timestamp}/', exist_ok=True)

    average_spending = df['price'].mean()
    average_spending_by_generation = df.groupby('social_generation')['price'].mean()

    meme_change = coin_data['meme_change']

    generations = average_spending_by_generation.index.tolist()

    generation_spending_pct = ((average_spending_by_generation - average_spending) / average_spending) * 100
    generation_spendings = np.array(generation_spending_pct, dtype=float)

    generation_spendings_meme_correlation = generation_spendings * meme_change

    plt.figure(figsize=(10, 6))
    plt.bar(generations, generation_spendings_meme_correlation)
    plt.axhline(0, color='gray', linestyle='--')

    plt.title('Meme Coins Change vs Generation Spendings', fontsize=14)
    plt.ylabel('Spending x Meme Coins Change Factor', fontsize=12)
    plt.xlabel('Social Generation', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'/images/{timestamp}/generation_meme_coins_correlation.png')
    plt.close()
    logging.info("Plot saved to /generation_meme_coins_correlation.png")

# Looking for a correlation between Trump-endorsed/Trump-created coin performance and spending habits for different state colors
# 24h trump coins change% * state color spending% (compared to the average)
@task
def plot_state_color_trump_coins_correlation(tbl_dict, coin_data):
    df = pd.DataFrame.from_dict(tbl_dict)
    timestamp = coin_data["timestamp"]

    os.makedirs('/images/', exist_ok=True)
    os.makedirs(f'/images/{timestamp}/', exist_ok=True)

    average_spending = df['price'].mean()
    average_spending_by_state_color = df.groupby('state_color')['price'].mean()

    trump_change = coin_data['trump_change']

    state_colors = average_spending_by_state_color.index.tolist()

    state_color_spending_pct = ((average_spending_by_state_color - average_spending) / average_spending) * 100
    state_color_spendings = np.array(state_color_spending_pct, dtype=float)

    state_color_spendings_trump_correlation = state_color_spendings * trump_change

    plt.figure(figsize=(10, 6))
    plt.bar(state_colors, state_color_spendings_trump_correlation)
    plt.axhline(0, color='gray', linestyle='--')

    plt.title('Trump Coins Change vs Red/Blue State Spendings', fontsize=14)
    plt.ylabel('Spending x Trump Coins Change Factor', fontsize=12)
    plt.xlabel('State Color', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'/images/{timestamp}/state_color_trump_coins_correlation.png')
    plt.close()
    logging.info("Plot saved to /state_color_trump_coins_correlation.png")

# Looking for a correlation between RWA (Real World Assets) coin performance and consumers choosing to spend more on accessories or clothing
# 24h RWA coins change% * average accesories/clothing spending%
# Calculating the average spending on accessories/clothing by every consumer (but consumers are unique in this dataset so it won't make any difference)
@task
def plot_category_rwa_coins_correlation(tbl_dict, coin_data):
    df = pd.DataFrame.from_dict(tbl_dict)
    timestamp = coin_data["timestamp"]

    os.makedirs('/images/', exist_ok=True)
    os.makedirs(f'/images/{timestamp}/', exist_ok=True)

    average_spending = df['price'].mean()

    totals = df.groupby("category")["price"].sum()
    totals = totals.loc[["Clothing", "Accessories"]]

    n_customers = df['customer_id'].nunique()

    average_accessories_clothing_spending = totals.apply(lambda x: x / n_customers)

    rwa_change = coin_data['rwa_change']

    categories = average_accessories_clothing_spending.index.tolist()

    accesories_clothing_spending_pct = ((average_accessories_clothing_spending - average_spending) / average_spending) * 100
    accesories_clothing_spendings = np.array(accesories_clothing_spending_pct, dtype=float)

    accesories_clothing_spendings_rwa_correlation = accesories_clothing_spendings * rwa_change

    plt.figure(figsize=(10, 6))
    plt.bar(categories, accesories_clothing_spendings_rwa_correlation)
    plt.axhline(0, color='gray', linestyle='--')

    plt.title('RWA Coins Change vs Accesories/Clothing Spendings', fontsize=14)
    plt.ylabel('Spending x RWA Coins Change Factor', fontsize=12)
    plt.xlabel('Category', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'/images/{timestamp}/accesories_clothing_RWA_coins_correlation.png')
    plt.close()
    logging.info("Plot saved to /accesories_clothing_RWA_coins_correlation.png")

# Looking for a correlation between Global coin performance and price points that consumers choose to spend on
# 24h RWA coins change% * average purchase category spending%
# Calculating the average spending on every purchase price point category for every consumer
# (but consumers are unique in this dataset so it won't make any difference)
@task
def plot_price_points_global_coins_correlation(tbl_dict, coin_data):
    df = pd.DataFrame.from_dict(tbl_dict)
    timestamp = coin_data["timestamp"]

    os.makedirs('/images/', exist_ok=True)
    os.makedirs(f'/images/{timestamp}/', exist_ok=True)

    average_spending = df['price'].mean()

    totals = df.groupby("amount_category")["price"].sum()
    n_customers = df['customer_id'].nunique()

    average_purchase_category_spending = totals.apply(lambda x: x / n_customers)

    global_change = coin_data['global_change']

    categories = average_purchase_category_spending.index.tolist()

    purchase_category_spending_pct = ((average_purchase_category_spending - average_spending) / average_spending) * 100
    purchase_category_spendings = np.array(purchase_category_spending_pct, dtype=float)

    purchase_category_spendings_global_correlation = purchase_category_spendings * global_change

    plt.figure(figsize=(10, 6))
    plt.bar(categories, purchase_category_spendings_global_correlation)
    plt.axhline(0, color='gray', linestyle='--')

    plt.title('Global Market Change vs Price Point Spendings', fontsize=14)
    plt.ylabel('Spending x Global Change Factor', fontsize=12)
    plt.xlabel('Category', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'/images/{timestamp}/purchase_category_global_coins_correlation.png')
    plt.close()
    logging.info("Plot saved to /purchase_category_global_coins_correlation.png")

@task
def load_data_to_db_shopping(tbl_dict):
    # logging.info(f"Connecting to database at postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}")
    engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}")
    df = pd.DataFrame.from_dict(tbl_dict)

    logging.info(f"Dataframe columns: {df.columns.tolist()}")

    # Check for existing customer IDs to avoid duplicates and existing ids error
    existing_customers = pd.read_sql("SELECT customer_id FROM customers", engine)
    existing_customers_ids = set(existing_customers['customer_id'])

    logging.info(f"Existing customer IDs in database: {existing_customers_ids}")

    new_customer_rows = df[~df['customer_id'].isin(existing_customers_ids)]

    logging.info(f"New customer rows to be added: {len(new_customer_rows)}")

    if not new_customer_rows.empty:
        df = new_customer_rows.copy()
        dim_customer = df[["customer_id", "age", "gender", "location", "state_color", "social_generation", "subscription_status",
                        "purchase_frequency", "previous_purchases"]].drop_duplicates().reset_index(drop=True)

        dim_item = df[["item_purchased", "category", "size", "color", "rating"]].drop_duplicates().reset_index(drop=True)
        dim_item.rename(columns={"item_purchased": "item_name"}, inplace=True)
        dim_item["item_id"] = range(1, len(dim_item) + 1)

        existing_items = pd.read_sql("SELECT item_id FROM items", engine)
        existing_items_ids = set(existing_items['item_id'])
        dim_item = dim_item[~dim_item['item_id'].isin(existing_items_ids)]

        dim_receipt = df[["discount_applied", "promo_code_used", "amount_category", "payment_method"]].drop_duplicates().reset_index(drop=True)
        dim_receipt["receipt_id"] = range(1, len(dim_receipt) + 1)

        existing_receipts = pd.read_sql("SELECT receipt_id FROM receipts", engine)
        existing_receipt_ids = set(existing_receipts['receipt_id'])
        dim_receipt = dim_receipt[~dim_receipt['receipt_id'].isin(existing_receipt_ids)]

        fact_purchases = df.merge(dim_customer, on=["customer_id"])
        fact_purchases = fact_purchases.merge(dim_item, left_on=["item_purchased", "category", "size", "color", "rating"],
                                            right_on=["item_name", "category", "size", "color", "rating"])
        fact_purchases = fact_purchases.merge(dim_receipt, on=["discount_applied", "promo_code_used", "amount_category", "payment_method"])
        fact_purchases = fact_purchases[["customer_id", "item_id", "receipt_id", "price", "shipping_type", "season"]]
        fact_purchases["purchase_id"] = range(1, len(fact_purchases) + 1)

        dim_customer.to_sql('customers', engine, if_exists='append', index=False, method='multi', chunksize=1000)
        dim_item.to_sql('items', engine, if_exists='append', index=False, method='multi', chunksize=1000)
        dim_receipt.to_sql('receipts', engine, if_exists='append', index=False, method='multi', chunksize=1000)

        fact_purchases.to_sql('purchases', engine, if_exists='append', index=False, method='multi', chunksize=1000)

        logging.info("Data saved to dimension and fact tables in the database.")

        data = {
            "dim_customer_rows": len(dim_customer),
            "dim_item_rows": len(dim_item),
            "dim_receipt_rows": len(dim_receipt),
            "fact_purchases_rows": len(fact_purchases),
        }
    else:
        logging.info("No new customer data to load.")
        data = {
            "dim_customer_rows": 0,
            "dim_item_rows": 0,
            "dim_receipt_rows": 0,
            "fact_purchases_rows": 0,
        }

    return data

@task
def load_data_to_db_coins(data):
    logging.info(f"Connecting to database at postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}")
    engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}")
    df = pd.DataFrame([data])
    df.to_sql('financial_entries', engine, if_exists='append', index=False)
    logging.info("Data saved to coin_market_changes table in the database.")
    return data

with DAG(**cfg) as dag:

    @task_group
    def extract_data():
        shopping_data = fetch_data_shopping()
        coin_data = fetch_data_coins()

        return shopping_data, coin_data

    @task_group
    def transform_data(shopping_data, coin_data):
        shopping_data_renamed = rename_columns(shopping_data)
        shopping_data_full = compute_extra_columns(shopping_data_renamed)
        coin_data_clean = clean_coin_data(coin_data)

        plot_generation_meme_coins_correlation(shopping_data_full, coin_data_clean)
        plot_state_color_trump_coins_correlation(shopping_data_full, coin_data_clean)
        plot_category_rwa_coins_correlation(shopping_data_full, coin_data_clean)
        plot_price_points_global_coins_correlation(shopping_data_full, coin_data_clean)

        return shopping_data_full, coin_data_clean

    @task_group
    def load_data(shopping_data_full, coin_data):
        load_shopping = load_data_to_db_shopping(shopping_data_full)
        logging.info(f"New shopping data sizes: {load_shopping}")
        load_coins = load_data_to_db_coins(coin_data)

        return load_shopping, load_coins

    shopping_data, coin_data = extract_data()
    shopping_data_full, coin_data_clean = transform_data(shopping_data, coin_data)
    load_shopping, load_coins = load_data(shopping_data_full, coin_data_clean)
