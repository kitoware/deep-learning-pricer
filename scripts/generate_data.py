from QuantLib import *
import pandas as pd
import numpy as np

def generate_asian_data():
    S0_values = np.linspace(80, 120, 20)  
    sigma_values = np.linspace(0.1, 0.6, 30)  
    T_values = np.linspace(0.1, 1.0, 10)  
    data = []

    calendar = TARGET()
    day_count = Actual360()
    today = Date.todaysDate()
    Settings.instance().evaluationDate = today
    risk_free_rate = 0.05

    for S0 in S0_values:
        for sigma in sigma_values:
            for T in T_values:
                maturity_date = calendar.advance(today, int(T * 360), Days)
                
                K_values = [0.8 * S0, S0, 1.2 * S0]  

                for K in K_values:
                    spot = QuoteHandle(SimpleQuote(S0))
                    dividend_yield = YieldTermStructureHandle(FlatForward(0, calendar, 0.0, day_count))
                    risk_free_ts = YieldTermStructureHandle(FlatForward(0, calendar, risk_free_rate, day_count))
                    vol_ts = BlackVolTermStructureHandle(BlackConstantVol(0, calendar, sigma, day_count))

                    bs_process = GeneralizedBlackScholesProcess(spot, dividend_yield, risk_free_ts, vol_ts)

                    mc_engine = MCDiscreteArithmeticAPEngine(bs_process, "lowdiscrepancy", False, True, True, 10000)

                    num_fixings = int(T * 12)  
                    fixing_dates = [calendar.advance(today, i * int(T * 30), Days) for i in range(1, num_fixings + 1)]

                    payoff = PlainVanillaPayoff(Option.Call, K)
                    exercise = EuropeanExercise(maturity_date)
                    asian_option = DiscreteAveragingAsianOption(Average.Arithmetic, fixing_dates, payoff, exercise)
                    asian_option.setPricingEngine(mc_engine)

                    price = asian_option.NPV()
                    data.append([S0, risk_free_rate, T, K, sigma, price])

    df = pd.DataFrame(data, columns=["S0", "r", "T", "K", "sigma", "price"])
    df.to_csv("./data/generated_asian_option_data.csv", index=False)
    print("Data saved to ./data/generated_asian_option_data.csv")

if __name__ == "__main__":
    generate_asian_data()
