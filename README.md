
# 📈 Cryptocurrency Volatility Forecasting & Portfolio Optimization

This project applies GARCH(1,1) and EGARCH(1,1) models to Bitcoin return data for volatility forecasting and portfolio optimization.

---

## 📁 Project Structure

```
project-folder/
│
├── main.py                                                     # Main entry point of the project
├── requirements.txt                                            # Python dependencies
├── /data                                                       # Directory containing input datasets
├── model.py                                                    # GARCH and EGARCH model definitions
├── portofolio_optimization_and_evaluation.py                   # Portfolio optimization function
└── rolling_window_volatility.py                                # rolling window function
```

---

## ▶️ How to Run the Project

### 1. ✅ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### 2. 🐍 Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
```

---

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. 🚀 Run the Script

```bash
python main.py
```

> Make sure you are in the root directory where `main.py` is located when running the command.

---

## 🧪 Testing & Output

After running `main.py`, you will find:
- Forecasting metrics (MSE, RMSE, etc.)
- Portfolio performance metrics (Sharpe Ratio, Sortino Ratio, etc.)

---

## 🧰 Requirements

- Python 3.9+
- Libraries used:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `arch`
  - `scikit-learn`
  - `statsmodels`

---

## 📝 Notes

- Ensure the dataset (`.csv` or `.xlsx`) is located in the `/data` folder.
- Configuration variables (e.g., rolling window size, date range) may be set at the top of `main.py`.

---

## 📬 Questions or Issues?

Feel free to open an issue or contact the developer.
