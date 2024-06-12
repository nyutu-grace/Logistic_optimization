import pandas as pd
import holidays


# Load the dataset
completed_df = pd.read_csv('../data/nb.csv')

# Convert timestamp to datetime and handle errors
completed_df['datetime'] = pd.to_datetime(completed_df['timestamp'], errors='coerce')

# Drop rows with invalid datetime entries
completed_df = completed_df.dropna(subset=['datetime'])

# Initialize holidays for Nigeria
nigeria_holidays = holidays.CountryHoliday('NG')

# Create a function to check for holidays
def is_holiday(date):
    return date in nigeria_holidays

# Apply the function to check if the date is a holiday
completed_df['is_holiday'] = completed_df['datetime'].apply(is_holiday)

# Display the DataFrame
print(completed_df)
