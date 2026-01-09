import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("food_dataset.csv")

# -----------------------------
# User Input
# -----------------------------
age = int(input("Enter Age: "))
weight = float(input("Enter Weight (kg): "))
height = float(input("Enter Height (cm): "))
activity_level = input("Activity Level (low/medium/high): ")
diet_preference = input("Diet Preference (Veg/Non-Veg): ")

# -----------------------------
# Nutrition Science Calculations
# -----------------------------
# BMI Calculation
height_m = height / 100
bmi = weight / (height_m ** 2)

# BMR (Mifflin-St Jeor Equation)
bmr = 10 * weight + 6.25 * height - 5 * age + 5

# Activity Multiplier
activity_map = {
    "low": 1.2,
    "medium": 1.55,
    "high": 1.9
}

tdee = bmr * activity_map.get(activity_level.lower(), 1.2)

# -----------------------------
# Filter Food Based on Preference
# -----------------------------
filtered_food = data[data["DietType"].str.lower() == diet_preference.lower()]

# -----------------------------
# Machine Learning Model
# -----------------------------
X = filtered_food[["Calories", "Protein", "Carbs", "Fat"]]

kmeans = KMeans(n_clusters=3, random_state=42)
filtered_food["Cluster"] = kmeans.fit_predict(X)

# Select Balanced Diet Cluster
balanced_cluster = filtered_food.groupby("Cluster")["Calories"].mean().idxmin()
recommended_food = filtered_food[filtered_food["Cluster"] == balanced_cluster]

# -----------------------------
# Output Results
# -----------------------------
print("\n--- HEALTH REPORT ---")
print(f"BMI: {bmi:.2f}")
print(f"Daily Calorie Requirement: {int(tdee)} kcal")

print("\n--- RECOMMENDED MEAL PLAN ---")
print(recommended_food[["Food", "Calories", "Protein", "Carbs", "Fat"]])

