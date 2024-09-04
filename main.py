import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import re
from fpdf import FPDF

# 1. Data Processing
def load_data():
    data = pd.read_csv('olympics2024.csv')
    df = pd.DataFrame(data)
    return df
  
def preprocess_data(df):
    label_encoder = LabelEncoder()
    if 'Country' in df.columns and 'Country Code' in df.columns:
        df['Country'] = label_encoder.fit_transform(df['Country'])
        df['Country'].fillna(df['Country'].mean())
        df['Country Code'] = label_encoder.fit_transform(df['Country Code'])
        df['Country Code'].fillna(df['Country Code'].mean())

    return df

# 2. Analysis Engine
def run_analysis(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(X)
    print(y)
 # Example: Linear Regression
    try:
        model_lr = LinearRegression()
        model_lr.fit(X, y)
        plt.figure(figsize=(8, 6))
        plt.scatter(X.iloc[:, 0], y, color='blue', label='Actual')
        plt.plot(X.iloc[:, 0], model_lr.predict(X), color='red', label='Predicted')
        plt.title("Linear Regression")
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.legend()
        plt.savefig(r"./linear_regression_plot.png")
        plt.close()
        print("Linear Regression Model Score:", model_lr.score(X, y))
    except Exception as e:
        print(f"Error running Linear Regression: {e}")
        print("Linear Regression Model Score:", model_lr.score(X, y))
    except Exception as e:
        print(f"Error running Linear Regression: {e}")
 # Example: K-Means Clustering
    try:
        model_km = KMeans(n_clusters=3,n_init=10)
        model_km.fit(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=model_km.labels_, cmap='viridis')
        plt.scatter(model_km.cluster_centers_[:, 0], model_km.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
        plt.title("K-Means Clustering")
        plt.xlabel("Medals")
        plt.ylabel("Country")
        plt.legend()
        plt.savefig(r"./kmeans_plot.png")
        plt.close()
        print("K-Means Score:", model_km.cluster_centers_)
    except Exception as e:
        print(f"Error running K-Means: {e}")

 # Example: Decision Tree
    try:
    
        model_dt = DecisionTreeClassifier()
        model_dt.fit(X, y)
        plt.figure(figsize=(8, 6))
        feature_importances = model_dt.feature_importances_
        indices = np.argsort(feature_importances)[::-1]
        plt.barh(range(X.shape[1]), feature_importances[indices], align='center')
        plt.yticks(range(X.shape[1]), [X.columns[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title("Decision Tree")
        plt.savefig(r"./decision_tree_plot.png")
        plt.close()
        print("Decision Tree Model Score:", model_dt.score(X, y))
    except Exception as e:
        print(f"Error running Decision Tree: {e}")

# 3. Report Generation
def generate_report(df):
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Country Code', y='Total', data=df)
        plt.title('Total Medals by Country')
        plt.xticks(fontsize=5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(r"./medal_report.png")
        plt.close()

        pdf = FPDF()
        pdf.add_page()

        
        # Insert generated graphs
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Graphs:", ln=True, align="C")
        
        pdf.image(r"./medal_report.png")
        pdf.ln(90)
        
        pdf.image(r"./linear_regression_plot.png", x=10, y=130, w=180)
        pdf.ln(90)
        pdf.image(r"./kmeans_plot.png", x=10, y=230, w=180)
        pdf.ln(90)
        pdf.image(r"./decision_tree_plot.png", x=10, y=330, w=180)
        


        with open("data", "w") as f:
            try:
                f.write("Olympics 2024 Data Analysis Report\n")
                f.write("Summary Statistics:\n")
                f.write(df.describe().to_string())
            except Exception as e:
                print(f"Error writing summary statistics: {e}")   
            try:    
                df = df.drop(columns=['Country', 'Country Code']) 
                f.write("\n\nCorrelation Matrix:\n")
                f.write(df.corr().to_string())
            except Exception as e:
                print(f"Error writing correlation matrix: {e}")

        print("Report generated successfully")
    except Exception as e:
        print(f"Error generating report: {e}")

# 4. User Interaction
def main():
    print("Hey, Welcome to the AI Employee for Data Analysis")
    print("You can ask me to load data, preprocess it, run analysis, or generate a report.\n Press 1 to load data\n Press 2 to preprocess the data\n Press 3 to generate the report\n Press 4 to generate report\n")
    print("Type 'exit' to quit.")

    df = None

    while True:
        user_input = input("\nWhat would you like to do? ")

        if user_input.lower() == "exit":
            print("Tata!")
            break

        elif re.search(r'1', user_input, re.IGNORECASE):
            df = load_data()
            if df is not None:
                print("Data loaded successfully.")
            else:
                print("Failed to load data.")

        elif re.search(r'2', user_input, re.IGNORECASE):
            if df is not None:
                df = preprocess_data(df)
                print("Data preprocessing completed.")
            else:
                print("Please load data first.")

        elif re.search(r'3', user_input, re.IGNORECASE):
            if df is not None:
                run_analysis(df)
            else:
                print("Please load and preprocess data first.")

        elif re.search(r'4', user_input, re.IGNORECASE):
            df = load_data()
            if df is not None:
                generate_report(df)
            else:
                print("Please load and preprocess data first.")

        else:
            print("Sorry, I didn't understand this command.")

if __name__ == "__main__":
    main()
