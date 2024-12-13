# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:14:02 2022

@author: ut34u3
"""

import numpy as np
import pandas as pd
import ssas_api
from datetime import datetime
import f



conn ="Data Source=ASTMBIMUT01PRD\AST_MBIMUT01_PRD; Catalog=CMIT; Integrated Security=SSPI"


issuer = 'Seche Environnement SA'
issuer = 'Tereos Fin Grp I'
dax_string_spread = '''
// DAX Query
DEFINE
  VAR __DS0FilterTable = 
    TREATAS({"''' + issuer + '''"}, 'DimAssetIboxxIssuer'[IssuerLabel])
  VAR __DS0FilterTable2 = 
    TREATAS({"No"}, 'DimCalendar'[IsWeekEnd])

  VAR __DS0FilterTable3 = 
    TREATAS({BLANK()}, 'FactRFQ'[RFQPotentialError5%])
    
  VAR __DS0FilterTable4 = 
    TREATAS({BLANK()}, 'FactRFQ'[RFQPotentialError5%])

  VAR __DS0Core = 
    SUMMARIZECOLUMNS(
      ROLLUPADDISSUBTOTAL(
        ROLLUPGROUP('DimAssetIboxx'[isin], 'DimCalendar'[Date], 'DimAssetIboxxIssuer'[IssuerLabel]), "IsGrandTotalRowTotal"
      ),
      __DS0FilterTable,
      __DS0FilterTable2,
      __DS0FilterTable3,
      "SumSpread", CALCULATE(SUM('FactBondPrice'[Spread]))
    )

  VAR __DS0PrimaryWindowed = 
    TOPN(
      502000,
      __DS0Core,
      [IsGrandTotalRowTotal],
      0,
      'DimCalendar'[Date],
      1,
      'DimAssetIboxx'[isin],
      1,
      'DimAssetIboxxIssuer'[IssuerLabel],
      1
    )

EVALUATE
  __DS0PrimaryWindowed

ORDER BY
  [IsGrandTotalRowTotal] DESC,
  'DimCalendar'[Date],
  'DimAssetIboxx'[isin],
  'DimAssetIboxxIssuer'[IssuerLabel]

'''
dfspread = ssas_api.get_DAX(connection_string=conn, dax_string=dax_string_spread)

del dfspread['[IsGrandTotalRowTotal]']
dfspread=dfspread.iloc[1:,:]
dfspread.columns = ['isin', 'date', 'issuer', 'spread']

df = pd.DataFrame(index=sorted(list(set(dfspread.date))), columns=set(dfspread['isin']))

for i in df.columns:
    a = f.process_isin(dfspread[dfspread['isin']==i])
    for j in a.index:
        df[i][a.date[j]] = a.variation[j]

df = df.iloc[1:-1,:]
dff = pd.DataFrame(df.mean(axis=1).dropna(), columns=['spread'])

df['count'] = 0
for i in range(df.shape[0]):
    df['count'].iloc[i] = df.iloc[i,:].count()



l='{'
for i in list(set(dfspread['isin'])):
    l = l + '"' + i + '",'
l=l[:-1] + '}'

dax_string_rfq = """
// DAX Query
DEFINE
  MEASURE 'FactRFQ'[DealQty] = 
    (/* USER DAX BEGIN */
[DealNominalQty for Buy] - [DealNominalQty for Sell]
/* USER DAX END */)

  MEASURE 'FactRFQ'[DealNominalQty for Buy] = 
    (/* USER DAX BEGIN */

CALCULATE(
	SUM('FactRFQ'[DealNominalQty]),
	'FactRFQ'[VerbClient] IN { "Buy" }
)
/* USER DAX END */)

  MEASURE 'FactRFQ'[DealNominalQty for Sell] = 
    (/* USER DAX BEGIN */

CALCULATE(
	SUM('FactRFQ'[DealNominalQty]),
	'FactRFQ'[VerbClient] IN { "Sell" }
)
/* USER DAX END */)

  VAR __DS0FilterTable = 
    FILTER(
      KEEPFILTERS(VALUES('DimClient-Contract'[ClientSector])),
      NOT('DimClient-Contract'[ClientSector] = "Central Banks")
    )

  VAR __DS0FilterTable2 = 
    TREATAS("""+l+""", 'DimAssetIboxx'[isin])

  VAR __DS0FilterTable3 = 
    TREATAS({"No"}, 'DimCalendar'[IsWeekEnd])

  VAR __DS0FilterTable4 = 
    TREATAS({BLANK()}, 'FactRFQ'[RFQPotentialError5%])

  VAR __DS0Core = 
    SUMMARIZECOLUMNS(
      ROLLUPADDISSUBTOTAL('DimCalendar'[Date], "IsGrandTotalRowTotal"),
      __DS0FilterTable,
      __DS0FilterTable2,
      __DS0FilterTable3,
      __DS0FilterTable4,
      "DealQty", 'FactRFQ'[DealQty]
    )

  VAR __DS0PrimaryWindowed = 
    TOPN(502000, __DS0Core, [IsGrandTotalRowTotal], 0, 'DimCalendar'[Date], 1)

EVALUATE
  __DS0PrimaryWindowed

ORDER BY
  [IsGrandTotalRowTotal] DESC, 'DimCalendar'[Date]
"""
dfrfq = ssas_api.get_DAX(connection_string=conn, dax_string=dax_string_rfq)
dfrfq = dfrfq.iloc[1:,:]
del dfrfq['[IsGrandTotalRowTotal]']
dfrfq.columns = ['date', 'rfq']

dfrfq.set_index('date', inplace=True)

for i in dfrfq.index:
    if i in df.index:
        dfrfq['rfq'][i] = dfrfq['rfq'][i]/df['count'][i]
    else:
        dfrfq['rfq'][i] = 0


df = pd.merge(dff, dfrfq, right_index=True, left_index=True, how='outer')
df.dropna(subset=['spread'], inplace=True)
df.fillna(0, inplace=True)

df.to_excel('spread.xlsx')








