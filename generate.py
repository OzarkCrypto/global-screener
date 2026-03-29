"""Global Screener — Multi-market 52-week high scanner with momentum scoring"""
import pandas as pd, numpy as np, yfinance as yf, json, warnings, os, time, requests
from datetime import datetime, timedelta
from io import StringIO
warnings.filterwarnings('ignore')

END = datetime.now().strftime('%Y-%m-%d')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ═══ MARKET DEFINITIONS ═══
# Namu Securities supported exchanges: NYSE, NASDAQ, AMEX, TSE, HKEX, SSE/SZSE (Connect), XETRA, LSE
MARKETS = {
    'US': {'name': 'United States', 'name_ko': '미국', 'currency': '$', 'suffix': '', 'source': 'yfinance'},
    'KR': {'name': 'South Korea', 'name_ko': '한국', 'currency': '₩', 'suffix': '', 'source': 'krx+naver'},
    'DE': {'name': 'Germany', 'name_ko': '독일', 'currency': '€', 'suffix': '.DE', 'source': 'yfinance'},
    'UK': {'name': 'United Kingdom', 'name_ko': '영국', 'currency': '£', 'suffix': '.L', 'source': 'yfinance'},
    'JP': {'name': 'Japan', 'name_ko': '일본', 'currency': '¥', 'suffix': '.T', 'source': 'yfinance'},
    'HK': {'name': 'Hong Kong', 'name_ko': '홍콩', 'currency': 'HK$', 'suffix': '.HK', 'source': 'yfinance'},
}

# ═══ THEME BASKETS (cross-market) ═══
THEMES = {
    'Nuclear/SMR': {
        'US': ['CEG','VST','CCJ','SMR','OKLO','NNE'],
        'KR': ['034020','052690'],  # Doosan Enerbility, KEPCO E&C
        'JP': ['9501.T'],  # TEPCO
        'UK': ['RR.L'],  # Rolls Royce
        'DE': [],
        'HK': ['1816.HK'],  # CGN Power
    },
    'Defense': {
        'US': ['LMT','RTX','NOC','GD','LHX','HII'],
        'KR': ['012450','079550','047810'],  # Hanwha Aero, LIG Nex1, KAI
        'DE': ['RHM.DE'],  # Rheinmetall
        'UK': ['BA.L'],  # BAE Systems
        'JP': ['7011.T','7012.T'],  # Mitsubishi HI, Kawasaki HI
        'HK': [],
    },
    'AI Infrastructure': {
        'US': ['NVDA','AVGO','ANET','VRT','DELL','SMCI','MRVL'],
        'KR': ['005930','000660'],  # Samsung, SK Hynix
        'DE': ['IFX.DE'],  # Infineon
        'UK': ['ARM.L'],  # ARM Holdings
        'JP': ['6857.T','8035.T'],  # Advantest, Tokyo Electron
        'HK': [],
    },
    'Robotics/Automation': {
        'US': ['ROK','TER','ABBNY','ISRG'],
        'KR': ['454910','277810'],  # Doosan Robotics, Rainbow Robotics
        'DE': ['KU2.DE'],  # Kuka (if traded)
        'UK': [],
        'JP': ['6954.T','6506.T'],  # Fanuc, Yaskawa
        'HK': [],
    },
    'Shipbuilding': {
        'US': [],
        'KR': ['329180','010140','042660'],  # HD HHI, Samsung Heavy, Hanwha Ocean
        'DE': [],
        'UK': [],
        'JP': ['7013.T'],  # IHI
        'HK': [],
    },
    'Tankers/Shipping': {
        'US': ['STNG','FRO','INSW','DHT','SBLK','GNK'],
        'KR': [],
        'DE': [],
        'UK': [],
        'JP': ['9104.T','9101.T'],  # Mitsui OSK, NYK
        'HK': [],
    },
    'Gold Miners': {
        'US': ['NEM','GOLD','AEM','WPM','FNV','GFI'],
        'KR': [],
        'DE': [],
        'UK': ['FRES.L','EDV.L'],  # Fresnillo, Endeavour
        'JP': [],
        'HK': ['2899.HK'],  # Zijin Mining
    },
    'EV/Battery': {
        'US': ['TSLA'],
        'KR': ['373220','006400','003670','247540'],  # LG Energy, Samsung SDI, POSCO Future M, Ecopro BM
        'DE': [],
        'UK': [],
        'JP': [],
        'HK': ['1211.HK','2594.HK'],  # BYD, BYD Electronic
    },
    'Cybersecurity': {
        'US': ['PANW','CRWD','ZS','FTNT','NET'],
        'KR': [],
        'DE': [],
        'UK': ['DRX.L'],  # Darktrace
        'JP': [],
        'HK': [],
    },
    'Luxury': {
        'US': ['RL'],
        'KR': [],
        'DE': [],
        'UK': ['BRBY.L'],  # Burberry
        'JP': [],
        'HK': [],
    },
    'Biotech/Pharma': {
        'US': ['LLY','NVO','AMGN','VKTX','HIMS'],
        'KR': ['207940','068270'],  # Samsung Bio, Celltrion
        'DE': [],
        'UK': ['AZN.L','GSK.L'],  # AstraZeneca, GSK
        'JP': ['4502.T','4519.T'],  # Takeda, Chugai
        'HK': [],
    },
}

# ═══ TICKER LISTS ═══
# US: Russell 3000 loaded from CSV
# KR: full KOSPI+KOSDAQ from KRX API
# Others: major indices constituents

def load_ticker_list(market):
    """Load ticker list for a market from CSV, return list of tickers."""
    path = os.path.join(DATA_DIR, f'{market.lower()}_tickers.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'ticker' in df.columns:
            return df['ticker'].tolist()
        elif 'symbol' in df.columns:
            return df['symbol'].tolist()
        return df.iloc[:,0].tolist()
    return []

def fetch_major_tickers_yf(market):
    """Fetch major tickers for a market via yfinance index constituents."""
    suffix = MARKETS[market]['suffix']
    tickers = []
    
    if market == 'DE':
        # DAX 40 + MDAX major
        dax = ['SAP.DE','SIE.DE','ALV.DE','DTE.DE','AIR.DE','MBG.DE','BMW.DE','BAS.DE','MUV2.DE',
               'IFX.DE','RHM.DE','HEN3.DE','ADS.DE','BEI.DE','DPW.DE','VNA.DE','HEI.DE',
               'SHL.DE','FRE.DE','MTX.DE','RWE.DE','EON.DE','DB1.DE','CON.DE','VOW3.DE',
               'PAH3.DE','MRK.DE','ZAL.DE','SY1.DE','P911.DE','DTG.DE','CBK.DE','HNR1.DE',
               'PUM.DE','LHA.DE','TKA.DE','FME.DE','1COV.DE','BNR.DE','QIA.DE']
        tickers = dax
    elif market == 'UK':
        # FTSE 100 major
        ftse = ['SHEL.L','AZN.L','HSBA.L','ULVR.L','BP.L','GSK.L','RIO.L','REL.L','DGE.L',
                'AAL.L','LSEG.L','CRH.L','BA.L','GLEN.L','NG.L','RR.L','LLOY.L','BARC.L',
                'PRU.L','CPG.L','EXPN.L','III.L','ABF.L','ANTO.L','SSE.L','NWG.L','FRES.L',
                'WPP.L','IHG.L','BT.A.L','JD.L','BRBY.L','DRX.L','EDV.L','DARK.L',
                'AHT.L','IMB.L','MNG.L','SVT.L','VOD.L','TSCO.L','BDEV.L','TW.L',
                'SGE.L','AUTO.L','SGRO.L','PSN.L','MNDI.L','PSON.L','SMT.L']
        tickers = ftse
    elif market == 'JP':
        # Nikkei 225 major
        nikkei = ['7203.T','6758.T','6861.T','8306.T','9432.T','6501.T','7267.T',
                  '9984.T','4502.T','6902.T','7751.T','6954.T','8035.T','6857.T',
                  '8316.T','6762.T','4063.T','9433.T','3382.T','2914.T','4568.T',
                  '7011.T','7012.T','9501.T','4519.T','6506.T','6503.T','7013.T',
                  '7974.T','6367.T','8058.T','8031.T','6301.T','8801.T','5401.T',
                  '4911.T','7269.T','7741.T','9104.T','9101.T','6702.T','6752.T',
                  '6723.T','8411.T','8766.T','4543.T','6098.T','3407.T','2502.T','2802.T']
        tickers = nikkei
    elif market == 'HK':
        # Hang Seng major
        hsi = ['0700.HK','9988.HK','0005.HK','1299.HK','2318.HK','0941.HK','1810.HK',
               '0388.HK','0027.HK','0003.HK','0011.HK','0016.HK','0883.HK','0857.HK',
               '1211.HK','2594.HK','0002.HK','1038.HK','0006.HK','0012.HK','0017.HK',
               '1816.HK','2899.HK','0386.HK','1928.HK','0001.HK','0066.HK','2388.HK',
               '0688.HK','1113.HK','0823.HK','0175.HK','2269.HK','9618.HK','9999.HK',
               '3690.HK','6098.HK','0968.HK','1024.HK','2382.HK','0241.HK','0285.HK']
        tickers = hsi
    
    return tickers

# ═══ KRX API ═══
KRX_API_KEY = '31C1A64E11714282BD3C0A3D1E17FC00D5965CA1'

def krx_api_fetch(endpoint, params):
    headers = {'AUTH_KEY': KRX_API_KEY}
    r = requests.get(f'https://data-dbg.krx.co.kr/svc/apis/{endpoint}', params=params, headers=headers, timeout=15)
    d = r.json()
    return d.get('OutBlock_1')

# ═══ COMMON STOCK FILTER ═══
_ETF_PFX = ['KODEX','TIGER','KBSTAR','ARIRANG','SOL ','HANARO','KOSEF','ACE ','TIMEFOLIO','PLUS ',
    'MASTER','BNK','VITA','KoAct','파인더','메리츠','미래에셋TIGER','RISE ','FOCUS','WOORI',
    'TRUE','TREX','KINDEX','QV ','1Q ','히어로','파워 ','마이다스','KIWOOM','WON']
_ETF_CTN = ['ETN','레버리지','인버스','선물','합성)','액티브','패시브','S&P','나스닥','MSCI','블룸버그',
    '채권','국채','회사채','금리','단기','머니마켓','CD금리','통안','자산배분','TDF','ELS','부동산']

def is_common_stock(name):
    n = str(name)
    if not n or len(n) < 2: return False
    for p in _ETF_PFX:
        if n.startswith(p): return False
    for c in _ETF_CTN:
        if c in n: return False
    if n.endswith('우') or n.endswith('우B') or n.endswith('우C') or n.endswith('우D'): return False
    if n.endswith('리츠') or '스팩' in n or 'SPAC' in n: return False
    return True

def is_junk_stock(name, ticker):
    n = str(name); nl = n.lower(); t = str(ticker).strip()
    if 'acquisition corp' in nl or 'blank check' in nl: return True
    if ('preferred' in nl or 'cumulative' in nl) and ('series' in nl or '%' in n): return True
    if '% notes due' in nl: return True
    if 'warrant' in nl and 'warranty' not in nl: return True
    if t.endswith('WW') or t.endswith('WS'): return True
    if 'contingent value right' in nl: return True
    return False

# ═══ DATA PIPELINE ═══
def load_or_create_csv(market):
    """Load existing daily CSV or return empty DataFrame."""
    path = os.path.join(DATA_DIR, f'{market.lower()}_daily.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  {market} CSV loaded: {len(df.columns)} tickers, {len(df)} days, last={df.index.max().strftime('%Y-%m-%d')}")
        return df
    return pd.DataFrame()

def save_csv(market, df):
    path = os.path.join(DATA_DIR, f'{market.lower()}_daily.csv')
    df.to_csv(path)

def update_kr_data(existing_df):
    """Update Korean stock data from KRX API."""
    today = pd.Timestamp.now().normalize()
    last_date = existing_df.index.max() if len(existing_df) > 0 else today - pd.Timedelta(days=365*2)
    
    if last_date >= today - pd.Timedelta(days=1):
        print(f"  KR data up to date")
        return existing_df
    
    print(f"  KR: fetching from KRX API since {last_date.strftime('%Y-%m-%d')}...")
    dt = last_date + pd.Timedelta(days=1)
    new_rows = []
    while dt <= today:
        dt_str = dt.strftime('%Y%m%d')
        all_data = []
        for ep in ['sto/stk_bydd_trd', 'sto/ksq_bydd_trd']:
            try:
                data = krx_api_fetch(ep, {'basDd': dt_str})
                if data: all_data.extend(data)
            except: pass
        if all_data and len(all_data) > 100:
            row = {}
            for r in all_data:
                code = str(r.get('ISU_SRT_CD', '')).strip()
                name = str(r.get('ISU_ABBRV', '')).strip()
                if not code or len(code) != 6 or not code.isdigit(): continue
                if not is_common_stock(name): continue
                try:
                    close = float(str(r.get('TDD_CLSPRC', '0')).replace(',',''))
                    vol = float(str(r.get('ACC_TRDVOL', r.get('TDD_TRDVOL', '0'))).replace(',',''))
                    if close > 0 and vol > 0: row[code] = close
                except: pass
            if row:
                new_rows.append((dt, row))
                if len(new_rows) % 10 == 0: print(f"    {dt_str}: {len(row)} stocks")
        dt += pd.Timedelta(days=1)
        time.sleep(0.3)
    
    if new_rows:
        new_df = pd.DataFrame([r for _, r in new_rows], index=[d for d, _ in new_rows])
        df = pd.concat([existing_df, new_df]).sort_index()
        df = df[~df.index.duplicated(keep='last')]
        print(f"  KR: +{len(new_rows)} trading days")
        return df
    return existing_df

def update_yf_data(market, existing_df, tickers):
    """Update data for yfinance-based markets."""
    if not tickers: return existing_df
    today = pd.Timestamp.now().normalize()
    last_date = existing_df.index.max() if len(existing_df) > 0 else today - pd.Timedelta(days=365*2)
    
    if last_date >= today - pd.Timedelta(days=2):
        print(f"  {market} data up to date")
        return existing_df
    
    start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    end = (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"  {market}: downloading {len(tickers)} tickers from {start}...")
    
    BATCH = 200
    new_frames = {}
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        try:
            raw = yf.download(batch, start=start, end=end, progress=False, group_by='ticker', threads=True)
            if raw is None or raw.empty: continue
            for t in batch:
                try:
                    if len(batch) == 1:
                        s = raw['Close'].squeeze()
                    else:
                        if t not in raw.columns.get_level_values(0): continue
                        s = raw[t]['Close'].squeeze()
                    if isinstance(s, pd.Series) and s.notna().sum() > 0:
                        new_frames[t] = s
                except: pass
        except: pass
        time.sleep(0.5)
    
    if new_frames:
        new_df = pd.DataFrame(new_frames)
        new_df.index = pd.to_datetime(new_df.index).tz_localize(None)
        df = pd.concat([existing_df, new_df]).sort_index()
        df = df[~df.index.duplicated(keep='last')]
        print(f"  {market}: +{len(new_df)} days, {len(new_frames)} tickers updated")
        return df
    return existing_df

# ═══ 52-WEEK HIGH + MOMENTUM COMPUTATION ═══
def compute_screener(market, df, name_map=None):
    """Compute 52-week highs + momentum scores for a market."""
    if df is None or df.empty or len(df) < 50:
        print(f"  {market}: insufficient data")
        return []
    
    # Ensure enough history
    min_days = min(252, len(df) - 1)
    
    results = []
    for ticker in df.columns:
        try:
            s = df[ticker].dropna()
            if len(s) < 50: continue
            
            cur = float(s.iloc[-1])
            if cur <= 0: continue
            
            # 52-week high/low
            lb = min(252, len(s) - 1)
            h52 = float(s.iloc[-lb:].max())
            l52 = float(s.iloc[-lb:].min())
            if h52 <= 0 or l52 <= 0: continue
            
            pct_from_high = (cur / h52 - 1) * 100
            pct_from_low = (cur / l52 - 1) * 100
            
            # Only include if within 3% of 52w high
            if pct_from_high < -3.0: continue
            
            # Momentum: 1M, 3M, 6M, 12M returns
            ret_1m = (cur / float(s.iloc[-min(21, len(s)-1)]) - 1) * 100 if len(s) > 21 else None
            ret_3m = (cur / float(s.iloc[-min(63, len(s)-1)]) - 1) * 100 if len(s) > 63 else None
            ret_6m = (cur / float(s.iloc[-min(126, len(s)-1)]) - 1) * 100 if len(s) > 126 else None
            ret_12m = (cur / float(s.iloc[-min(252, len(s)-1)]) - 1) * 100 if len(s) > 252 else None
            
            # Momentum score: weighted average of available returns
            scores = []
            if ret_3m is not None: scores.append(ret_3m * 0.3)
            if ret_6m is not None: scores.append(ret_6m * 0.4)
            if ret_12m is not None: scores.append(ret_12m * 0.3)
            mom_score = sum(scores) / max(1, len(scores)) if scores else 0
            
            # Volume trend (last 20d avg vs 60d avg)
            if len(s) > 60:
                vol_ratio = None  # will add volume data later
            
            name = name_map.get(ticker, ticker) if name_map else ticker
            
            entry = {
                't': ticker,
                'n': name,
                'p': round(cur, 2),
                'h52': round(h52, 2),
                'l52': round(l52, 2),
                'pctH': round(pct_from_high, 1),
                'pctL': round(pct_from_low, 1),
                'r1m': round(ret_1m, 1) if ret_1m is not None else None,
                'r3m': round(ret_3m, 1) if ret_3m is not None else None,
                'r6m': round(ret_6m, 1) if ret_6m is not None else None,
                'r12m': round(ret_12m, 1) if ret_12m is not None else None,
                'mom': round(mom_score, 1),
            }
            results.append(entry)
        except: pass
    
    # Sort by momentum score
    results.sort(key=lambda x: -(x['mom'] or 0))
    print(f"  {market}: {len(results)} stocks near 52-week high")
    return results

# ═══ THEME MATCHING ═══
def match_themes(all_results):
    """Match stocks to themes and compute theme-level signals."""
    theme_data = {}
    
    for theme_name, theme_tickers in THEMES.items():
        theme_stocks = {}
        for market, tickers in theme_tickers.items():
            if not tickers: continue
            market_results = all_results.get(market, [])
            # Find any stocks from this theme in the highs list
            highs_tickers = {r['t'] for r in market_results}
            for t in tickers:
                # Strip suffix for matching
                t_clean = t.replace('.T','').replace('.HK','').replace('.DE','').replace('.L','')
                matched = t in highs_tickers or t_clean in highs_tickers
                if matched:
                    stock = next((r for r in market_results if r['t'] == t or r['t'] == t_clean), None)
                    if stock:
                        if market not in theme_stocks: theme_stocks[market] = []
                        theme_stocks[market].append(stock)
        
        total_tickers = sum(len(v) for v in theme_tickers.values() if v)
        matched_count = sum(len(v) for v in theme_stocks.values())
        n_markets = len([m for m in theme_stocks if theme_stocks[m]])
        
        if matched_count > 0:
            theme_data[theme_name] = {
                'matched': matched_count,
                'total': total_tickers,
                'pct': round(matched_count / total_tickers * 100) if total_tickers > 0 else 0,
                'n_markets': n_markets,
                'markets': {m: stocks for m, stocks in theme_stocks.items()},
            }
    
    # Sort by cross-market breadth then match percentage
    return dict(sorted(theme_data.items(), key=lambda x: (-x[1]['n_markets'], -x[1]['pct'])))

# ═══ KR NAME MAP ═══
def build_kr_name_map():
    """Build Korean stock name map from KRX API or CSV."""
    path = os.path.join(DATA_DIR, 'kr_names.csv')
    name_map = {}
    
    # Load existing
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            name_map[str(r['code']).zfill(6)] = str(r['name'])
    
    # Try to update from KRX API
    try:
        data = krx_api_fetch('sto/stk_bydd_trd', {'basDd': datetime.now().strftime('%Y%m%d')})
        if not data:
            for delta in range(1, 10):
                dt = (datetime.now() - timedelta(days=delta)).strftime('%Y%m%d')
                data = krx_api_fetch('sto/stk_bydd_trd', {'basDd': dt})
                if data and len(data) > 100: break
                time.sleep(0.3)
        if data:
            for r in data:
                code = str(r.get('ISU_SRT_CD', '')).strip()
                name = str(r.get('ISU_ABBRV', '')).strip()
                if code and name and len(code) == 6: name_map[code] = name
            # Also fetch KOSDAQ
            for delta in range(0, 10):
                dt = (datetime.now() - timedelta(days=delta)).strftime('%Y%m%d')
                data2 = krx_api_fetch('sto/ksq_bydd_trd', {'basDd': dt})
                if data2 and len(data2) > 100:
                    for r in data2:
                        code = str(r.get('ISU_SRT_CD', '')).strip()
                        name = str(r.get('ISU_ABBRV', '')).strip()
                        if code and name and len(code) == 6: name_map[code] = name
                    break
                time.sleep(0.3)
            # Save
            pd.DataFrame([{'code': k, 'name': v} for k, v in name_map.items()]).to_csv(path, index=False)
            print(f"  KR names: {len(name_map)} stocks")
    except Exception as e:
        print(f"  KR names update failed: {e}")
    
    return name_map

# ═══ US NAME MAP ═══
def build_us_name_map():
    """Build US stock name + sector map."""
    path = os.path.join(DATA_DIR, 'us_ticker_list.csv')
    name_map = {}; sector_map = {}
    if os.path.exists(path):
        df = pd.read_csv(path)
        nm_col = 'name' if 'name' in df.columns else 'Company'
        tk_col = 'symbol' if 'symbol' in df.columns else 'Ticker'
        sc_col = 'sector' if 'sector' in df.columns else 'Sector'
        for _, r in df.iterrows():
            t = str(r.get(tk_col, '')).strip()
            if t:
                name_map[t] = str(r.get(nm_col, t))
                sector_map[t] = str(r.get(sc_col, ''))
    return name_map, sector_map

# ═══ MAIN PIPELINE ═══
print(f"Global Screener: {END}")

# 1. Load/update data per market
all_data = {}
all_results = {}

# Korea
print("\n=== KOREA ===")
kr_df = load_or_create_csv('KR')
kr_df = update_kr_data(kr_df)
if len(kr_df) > 0:
    save_csv('KR', kr_df)
    kr_names = build_kr_name_map()
    all_results['KR'] = compute_screener('KR', kr_df, kr_names)

# US
print("\n=== US ===")
us_df = load_or_create_csv('US')
us_tickers = load_ticker_list('US')
if not us_tickers:
    # Use broad universe from existing regime-monitor data
    us_tickers_path = os.path.join(DATA_DIR, 'us_ticker_list.csv')
    if os.path.exists(us_tickers_path):
        tl = pd.read_csv(us_tickers_path)
        tk_col = 'symbol' if 'symbol' in tl.columns else 'Ticker'
        us_tickers = tl[tk_col].dropna().tolist()
    else:
        # Fallback: use S&P 500 + Russell components from yfinance
        print("  No US ticker list found, using yfinance S&P 500")
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            us_tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
        except:
            us_tickers = ['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','AVGO','JPM','V']

us_df = update_yf_data('US', us_df, us_tickers)
if len(us_df) > 0:
    save_csv('US', us_df)
    us_names, us_sectors = build_us_name_map()
    us_results = compute_screener('US', us_df, us_names)
    # Add sector info
    for r in us_results:
        r['sector'] = us_sectors.get(r['t'], '')
    # Filter junk
    us_results = [r for r in us_results if not is_junk_stock(r['n'], r['t'])]
    all_results['US'] = us_results

# Europe + Asia
for market in ['DE', 'UK', 'JP', 'HK']:
    print(f"\n=== {MARKETS[market]['name'].upper()} ===")
    df = load_or_create_csv(market)
    tickers = load_ticker_list(market)
    if not tickers:
        tickers = fetch_major_tickers_yf(market)
        # Save ticker list
        if tickers:
            pd.DataFrame({'ticker': tickers}).to_csv(os.path.join(DATA_DIR, f'{market.lower()}_tickers.csv'), index=False)
    df = update_yf_data(market, df, tickers)
    if len(df) > 0:
        save_csv(market, df)
        all_results[market] = compute_screener(market, df)

# Theme matching
print("\n=== THEMES ===")
theme_results = match_themes(all_results)
for name, data in theme_results.items():
    markets = ', '.join(data['markets'].keys())
    print(f"  {name}: {data['matched']}/{data['total']} ({data['pct']}%) across {data['n_markets']} markets [{markets}]")

# ═══ BUILD OUTPUT ═══
output = {
    'date': END,
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
    'markets': {},
    'themes': {},
    'summary': {},
}

for market, results in all_results.items():
    output['markets'][market] = {
        'name': MARKETS[market]['name'],
        'name_ko': MARKETS[market]['name_ko'],
        'currency': MARKETS[market]['currency'],
        'count': len(results),
        'stocks': results[:500],
    }
    output['summary'][market] = len(results)

for name, data in theme_results.items():
    theme_out = {'matched': data['matched'], 'total': data['total'], 'pct': data['pct'],
        'n_markets': data['n_markets'], 'markets': {}}
    for m, stocks in data['markets'].items():
        theme_out['markets'][m] = [{'t': s['t'], 'n': s['n'], 'p': s['p'],
            'pctL': s['pctL'], 'r3m': s['r3m'], 'r6m': s['r6m'], 'mom': s['mom']} for s in stocks]
    output['themes'][name] = theme_out

data_str = json.dumps(output, separators=(',',':'), ensure_ascii=False)
print(f"\nOutput JSON: {len(data_str)/1024:.0f} KB")

# ═══ HTML ═══
FAVICON = 'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAAAAlwSFlzAAALEwAACxMBAJqcGAAAAVlpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KTMInWQAAA0pJREFUWAnFl0toE1EUhv8kk0naJi0tpFIEBUHc+AAXguBGxIWIG3GjuHPhwoULEQQRF6ILFy5EQdCVuhPBBxT3KoqKIj4QxBdaxaKitrVNm+bh//9OJp1MMpMqeOBm7p25539/7jl3bkwL/me4VIWl9Fx1dIR7B1NbLPsDLssFfYmYyH4qZ7LxGaH9D6eiQNrqD/p9fafnj2YmhLxeTwJ/5JOb7kYulJi5XGY26RnX9IxSxhkLZsPjkT5+bYuJRfJPHj+mNb3kX9K0L3k8kx8PU6KJqvxqkW6A0WUxibuS48F+N5YvCuIJcqhsS9InU3Fg5JNI7dCJ+kzaXlHSC5kDg=='

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Global Screener</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#fafaf9;--s:#fff;--b:#e7e5e4;--bl:#f5f5f4;--t:#1c1917;--t2:#57534e;--t3:#a8a29e;--g:#16a34a;--gb:#f0fdf4;--r:#dc2626;--rb:#fef2f2;--x:#78716c;--f:'DM Sans',sans-serif;--m:'JetBrains Mono',monospace}}
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:var(--f);background:var(--bg);color:var(--t);font-size:14px;line-height:1.5}}
.c{{max-width:1200px;margin:0 auto;padding:16px 14px}}
header{{display:flex;align-items:center;padding-bottom:8px;border-bottom:2px solid var(--t);margin-bottom:12px;gap:8px}}
header h1{{font-family:var(--m);font-size:16px;font-weight:800;letter-spacing:-0.03em}}header .dt{{font-size:11px;color:var(--t3);font-family:var(--m);margin-left:auto}}
.lt{{display:flex;gap:4px}}.lt button{{font-family:var(--m);font-size:12px;padding:6px 12px;border:1px solid var(--b);border-radius:6px;background:var(--s);color:var(--t3);cursor:pointer;font-weight:700;min-height:36px}}.lt button.on{{background:var(--t);color:var(--bg);border-color:var(--t)}}
.tabs{{display:flex;gap:0;margin-bottom:12px;border-radius:8px;overflow:hidden;border:2px solid var(--t)}}.tabs button{{flex:1;font-family:var(--m);font-size:13px;font-weight:700;padding:12px 8px;border:none;cursor:pointer;min-height:44px}}.tabs button.on{{background:var(--t);color:var(--bg)}}.tabs button:not(.on){{background:var(--bg);color:var(--t3)}}
.sm{{font-family:var(--m);font-size:10px;color:var(--t3);margin-bottom:10px;padding:6px 10px;background:var(--bl);border-radius:4px}}
.sort{{display:flex;gap:4px;margin-bottom:10px;flex-wrap:wrap}}.sort button{{font-family:var(--m);font-size:11px;padding:6px 10px;border:1px solid var(--b);border-radius:4px;background:var(--s);color:var(--t2);cursor:pointer;min-height:32px}}.sort button.on{{background:var(--t);color:var(--s);border-color:var(--t)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:6px;margin-bottom:16px}}
.card{{padding:10px 14px;border-radius:8px;background:var(--s);border:1px solid var(--b);border-left:4px solid var(--g)}}
.card .top{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px}}
.card .nm{{font-family:var(--m);font-size:13px;font-weight:700;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}}
.card .pr{{font-family:var(--m);font-size:15px;font-weight:800;text-align:right;flex-shrink:0;margin-left:8px}}
.card .sub{{font-family:var(--m);font-size:10px;color:var(--t3)}}
.card .mom{{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}}
.card .tag{{font-family:var(--m);font-size:10px;padding:2px 6px;border-radius:3px;font-weight:600}}
.tag.pos{{color:var(--g);background:var(--gb)}}.tag.neg{{color:var(--r);background:var(--rb)}}.tag.neu{{color:var(--x);background:var(--bl)}}
.thm{{margin-bottom:8px;padding:12px 16px;border-radius:10px;background:var(--s);border:1px solid var(--b)}}
.thm h3{{font-family:var(--m);font-size:13px;font-weight:800;margin-bottom:4px}}.thm .meta{{font-family:var(--m);font-size:10px;color:var(--t3)}}
.thm .stocks{{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px}}
.thm .st{{font-family:var(--m);font-size:10px;padding:3px 8px;border-radius:4px;background:var(--gb);color:var(--g);font-weight:600}}
.badge{{font-family:var(--m);font-size:9px;font-weight:700;padding:2px 6px;border-radius:3px;background:var(--t);color:var(--bg)}}
.empty{{padding:40px;text-align:center;color:var(--t3);font-family:var(--m);font-size:14px}}
footer{{margin-top:16px;padding-top:8px;border-top:1px solid var(--b);font-size:9px;color:var(--t3);text-align:center;font-family:var(--m)}}
@media(max-width:640px){{.grid{{grid-template-columns:1fr}}.tabs button{{font-size:11px;padding:10px 4px}}}}
</style></head><body><div class="c" id="app"></div>
<script>
const D={data_str};
const MK=['KR','US','DE','UK','JP','HK'];
const ML={{'KR':'\\ud55c\\uad6d','US':'US','DE':'DE','UK':'UK','JP':'JP','HK':'HK'}};
let S={{mkt:'ALL',sort:'mom',lang:'en',page:'markets'}};
function t(k){{return k}}
function render(){{
const app=document.getElementById('app');
let h='<header><h1>Global Screener</h1><div class="dt">'+D.generated+'</div><div class="lt">';
['en','ko'].forEach(l=>{{h+='<button class="'+(S.lang===l?'on':'')+'" onclick="S.lang=\\''+l+'\\';render()">'+(l==='en'?'EN':'\\ud55c')+'</button>'}});
h+='</div></header>';
h+='<div class="tabs"><button class="'+(S.page==='markets'?'on':'')+'" onclick="S.page=\\'markets\\';render()">'+(S.lang==='ko'?'\\uc2dc\\uc7a5\\ubcc4':'Markets')+'</button>';
h+='<button class="'+(S.page==='themes'?'on':'')+'" onclick="S.page=\\'themes\\';render()">'+(S.lang==='ko'?'\\ud14c\\ub9c8':'Themes')+'</button></div>';
if(S.page==='themes')h+=pgThemes();else h+=pgMarkets();
h+='<footer>Global 52-week high screener \\u00b7 KR + US + DE + UK + JP + HK \\u00b7 Daily auto-update</footer>';
app.innerHTML=h}}
function pgMarkets(){{
let h='<div class="tabs" style="border-width:1px">';
h+='<button class="'+(S.mkt==='ALL'?'on':'')+'" onclick="S.mkt=\\'ALL\\';render()">ALL</button>';
MK.forEach(m=>{{if(!D.markets[m])return;const cnt=D.markets[m].count;h+='<button class="'+(S.mkt===m?'on':'')+'" onclick="S.mkt=\\''+m+'\\';render()">'+ML[m]+' <span style="font-size:10px;opacity:.6">'+cnt+'</span></button>'}});
h+='</div>';
h+='<div class="sort">';
[['mom','Momentum'],['pctL','% from Low'],['r12m','12M Return'],['r6m','6M'],['r3m','3M'],['r1m','1M']].forEach(([k,v])=>{{
h+='<button class="'+(S.sort===k?'on':'')+'" onclick="S.sort=\\''+k+'\\';render()">'+v+'</button>'}});
h+='</div>';
let stocks=[];
const mkts=S.mkt==='ALL'?MK:[S.mkt];
mkts.forEach(m=>{{if(!D.markets[m])return;const cur=D.markets[m].currency;D.markets[m].stocks.forEach(s=>{{stocks.push(Object.assign({{}},s,{{mkt:m,cur:cur}}))}});}});
stocks.sort((a,b)=>(b[S.sort]||0)-(a[S.sort]||0));
const shown=stocks.slice(0,300);
h+='<div class="sm">'+shown.length+' of '+stocks.length+' stocks near 52-week high</div>';
h+='<div class="grid">';
shown.forEach(s=>{{
const tc=v=>v>0?'pos':v<0?'neg':'neu';
h+='<div class="card"><div class="top"><div style="min-width:0;flex:1"><div class="nm">'+s.n+'</div><div class="sub">'+s.t+(s.sector?' \\u00b7 '+s.sector:'')+' \\u00b7 <span class="badge">'+s.mkt+'</span></div></div>';
h+='<div class="pr">'+s.cur+s.p.toLocaleString()+'</div></div>';
h+='<div class="mom">';
if(s.pctL!=null)h+='<span class="tag pos">Low+'+s.pctL+'%</span>';
if(s.r1m!=null)h+='<span class="tag '+tc(s.r1m)+'">1M '+(s.r1m>0?'+':'')+s.r1m+'%</span>';
if(s.r3m!=null)h+='<span class="tag '+tc(s.r3m)+'">3M '+(s.r3m>0?'+':'')+s.r3m+'%</span>';
if(s.r6m!=null)h+='<span class="tag '+tc(s.r6m)+'">6M '+(s.r6m>0?'+':'')+s.r6m+'%</span>';
if(s.r12m!=null)h+='<span class="tag '+tc(s.r12m)+'">12M '+(s.r12m>0?'+':'')+s.r12m+'%</span>';
h+='</div></div>'}});
h+='</div>';return h}}
function pgThemes(){{
let h='';const themes=D.themes;
if(!themes||Object.keys(themes).length===0){{return'<div class="empty">No cross-market themes detected</div>'}}
const sorted=Object.entries(themes).sort((a,b)=>b[1].n_markets-a[1].n_markets||b[1].pct-a[1].pct);
sorted.forEach(([name,data])=>{{
const fire=data.n_markets>=3?'\\ud83d\\udd25 ':'';
h+='<div class="thm"><h3>'+fire+name+'</h3>';
h+='<div class="meta">'+data.matched+'/'+data.total+' stocks ('+data.pct+'%) across '+data.n_markets+' markets</div>';
h+='<div class="stocks">';
Object.entries(data.markets).forEach(([m,stocks])=>{{
stocks.forEach(s=>{{
h+='<span class="st">'+ML[m]+' '+s.n+' '+(s.r6m!=null?(s.r6m>0?'+':'')+s.r6m+'%':'')+'</span>'}});
}});
h+='</div></div>'}});
return h}}
render();
</script></body></html>"""

with open('index.html', 'w') as f:
    f.write(html)
print(f"OK index.html: {len(html)/1024:.1f} KB")
