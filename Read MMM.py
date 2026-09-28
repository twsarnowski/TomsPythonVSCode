from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


workbook_path = Path(__file__).resolve().parent.parent / "personal_computing_weekly_data_104_weeks_enhanced.xlsx"
dataframe = pd.read_excel(workbook_path, sheet_name="Weekly Data")
dataframe = dataframe.sort_values(by=["Week Start", "Product Segment"])
print(dataframe.head(5).to_string(index=False))

continuous_measures = [
	"Unit Sales",
	"Display Impressions",
	"Display Spend",
	"Average Price",
	"Inventory Out of Stock Percentage",
]
averages_by_product = dataframe.groupby("Product Segment")[continuous_measures].mean()
print("\nAverage continuous measures by product segment:")
print(averages_by_product.to_string(formatters={
	"Unit Sales": "{:,.0f}".format,
	"Display Impressions": "{:,.0f}".format,
	"Display Spend": "${:,.0f}".format,
	"Average Price": "${:,.2f}".format,	
    "Inventory Out of Stock Percentage": "{:,.4f}".format,
}))

weekly_sales = dataframe.pivot(index="Week Start", columns="Product Segment", values="Unit Sales")
ax = weekly_sales.plot(figsize=(11, 6), linewidth=1.8)
ax.set(title="Weekly Unit Sales by Product Segment", xlabel="Week", ylabel="Unit Sales")
ax.grid(axis="y", alpha=0.3)
ax.legend(title="Product Segment")
figure = ax.get_figure()
figure.tight_layout()
figure.savefig(Path(__file__).resolve().parent / "weekly_unit_sales_by_product.png", dpi=150)
plt.show()
