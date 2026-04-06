import numpy as np
import scipy
import math
import pandas as pd
from typing import Any
from numpy.typing import NDArray
from docx import Document
import plotly.graph_objects as go

# Вариант 11
N = 10
P = 0.351
SIZE = 200
ALPHA = 0.05
SEED = 1000+11


#моделирование 
def binom_dist(n: int, p: float) -> list[float]:
    return [round((math.comb(n, k) * p**k * (1 - p) ** (n - k)), 5) for k in range(n + 1)]


def stand_method(distribution: list[float], size: int) -> NDArray[np.int64]:
    np.random.seed(SEED)
    return np.searchsorted(np.cumsum(distribution), np.random.random(size))

#критерии оценки

def ksi_square(data_table: pd.DataFrame, size: int) -> pd.DataFrame:
    data = data_table[['w_i', 'p_i']].copy()
        
    data['|w_i-p_i|'] = round((data['w_i'] - data['p_i']).abs(), 5)
    
    data['N*(w_i-p_i)^2/p_i'] = round(size * (data['w_i'] - data['p_i']).abs()**2 / data['p_i'],5)
    
    return data

def uniformity_data(st_method_table: pd.DataFrame, scipy_method_table: pd.DataFrame):
    data = pd.DataFrame()

    data['w_i1'] = st_method_table['w_i'].copy()
    data['w_i2'] = scipy_method_table['w_i'].copy()

    denom = data['w_i1'] + data['w_i2']
    
    data['unif. criteria'] = round(
        (data['w_i1']**2 + data['w_i2']**2) / denom.replace(0, np.nan),
        5
    )

    return data


def theor_exper_analysis(distib: list[int], mean_theor:float, var_theor:float)-> pd.DataFrame:
    mean_exp = np.mean(distib)
    var_exp = np.var(distib, ddof=0)

    abs_mean = abs(mean_exp - mean_theor)
    abs_var = abs(var_exp - var_theor)

    rel_mean = round(abs_mean / abs(mean_theor), 5) if mean_theor != 0 else '–'
    rel_var = round(abs_var / abs(var_theor), 5) if var_theor != 0 else '–'

    return pd.DataFrame({
        "Название показателя": ["Выборочное среднее", "Выборочная дисперсия"],
        "Экспериментальное значение": [round(mean_exp, 5), round(var_exp, 5)],
        "Теоретическое значение": [round(mean_theor, 5), round(var_theor, 5)],
        "Абсолютное отклонение": [round(abs_mean, 5), round(abs_var, 5)],
        "Относительное отклонение": [rel_mean, rel_var]
    })



#формирование таблиц
def make_dicts_ans(
    ans: list[int], distr: list[float], value: int, size: int
) -> dict[str, list[Any] | NDArray[Any]]:
    counts = np.bincount(ans, minlength=value + 1)
    return {
        "x_i": list(range(value + 1)),
        "n_i": counts,
        "w_i": [round((i / size),5) for i in counts],
        "p_i": distr,
        "s_i": np.cumsum(distr),
    }

def data_table_20x10(distrib)-> pd.DataFrame:
    data_f = pd.DataFrame(np.array(distrib).reshape(20, 10))
    data_f.index = [''] * 20
    data_f.columns = [''] * 10
    return data_f


def table(data: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(data)

#конвертация в док

def add_df_to_doc(doc: Document, df: pd.DataFrame, title: str):
    doc.add_heading(title, level=2)
    
    table = doc.add_table(rows=df.shape[0] + 1, cols=df.shape[1])
    
    for j, col in enumerate(df.columns):
        table.rows[0].cells[j].text = str(col)
    
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            table.rows[i + 1].cells[j].text = str(df.iloc[i, j])
            
#графики

import plotly.graph_objects as go
import pandas as pd


def plot_polygons(
    stand_method: pd.DataFrame,
    scipy_method: pd.DataFrame,
    title: str = "Полигон относительных частот и вероятностей"
) -> None:
    """
    Строит:
    - полигон относительных частот (выборка 1)
    - полигон относительных частот (выборка 2)
    - полигон вероятностей (теоретический)
    """

    x = stand_method["x_i"]

    w1 = stand_method["w_i"]
    w2 = scipy_method["w_i"]
    p = stand_method["p_i"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=w1,
        mode='lines+markers',
        name='w_i (стандартный метод)',
        line=dict(width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=x, y=w2,
        mode='lines+markers',
        name='w_i (scipy)',
        line=dict(width=2, dash='dash'),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=x, y=p,
        mode='lines+markers',
        name='p_i (теоретическое)',
        line=dict(width=3),
        marker=dict(size=7)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="x_i",
        yaxis_title="Значение",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98)
    )

    fig.show()
#---------
binom = binom_dist(N, P)

# 1. Стандартный метод
ans = stand_method(binom, SIZE)
print("\n=== Стандартный метод ===")
table_1 =table(make_dicts_ans(ans, binom, N, SIZE))
print(f'\n Исходные данные \n{data_table_20x10(ans)}')
print(f'\n Исходные сортированные данные \n{data_table_20x10(sorted(ans))}')
print(f'\nСтатистические ряды\n{table_1}')
print(ksi_square(table_1, SIZE))
print(table_1.sum())
print(ksi_square(table_1, SIZE).sum())
print(theor_exper_analysis(ans, N*P, N*P*(1-P)))

# 2. scipy
sc = scipy.stats.binom.rvs(N, P, size=SIZE, random_state=SEED+1)
print("\n=== scipy ===")
table_2 = table(make_dicts_ans(sc, binom, N, SIZE))
print(f'\n Исходные данные \n{data_table_20x10(sc)}')
print(f'\n Исходные сортированные данные \n{data_table_20x10(sorted(sc))}')
print(f'\nСтатистические ряды\n{table_2}')
print(ksi_square(table_2, SIZE))
print(table_2.sum())
print(ksi_square(table_2, SIZE).sum())
print(theor_exper_analysis(sc, N*P, N*P*(1-P)))

plot_polygons(table_1, table_2)
doc = Document()

doc.add_heading('Отчет по моделированию', 0)

# 1. Стандартный метод
add_df_to_doc(doc, data_table_20x10(ans), "Исходные данные")
add_df_to_doc(doc, data_table_20x10(sorted(ans)), "Отсортированные данные")
add_df_to_doc(doc, table_1, "Статистический ряд")
add_df_to_doc(doc, ksi_square(table_1, SIZE), "Критерий хи-квадрат")

# 2. scipy 
add_df_to_doc(doc, data_table_20x10(sc), "Исходные данные (scipy)")
add_df_to_doc(doc, data_table_20x10(sorted(sc)), "Отсортированные данные (scipy)")
add_df_to_doc(doc, table_2, "Статистический ряд (scipy)")
add_df_to_doc(doc, ksi_square(table_2, SIZE), "Критерий хи-квадрат (scipy)")

# Сохранение
doc.save("report_1011.docx")

print(uniformity_data(table_1,table_2))
