import pandas as pd
import numpy as np
import json
from datetime import datetime
import requests

# Fetch the CSV data
url = 'https://www.football-data.co.uk/mmz4281/2526/E0.csv'
df = pd.read_csv(url)

# --- PRE-PROCESSING & HELPERS ---
def get_points(res, target):
    """Calculate points earned by a team based on the match result."""
    if res == 'D': return 1
    if res == target: return 3
    return 0

df['HomePts'] = df.apply(lambda r: get_points(r['FTR'], 'H'), axis=1)
df['AwayPts'] = df.apply(lambda r: get_points(r['FTR'], 'A'), axis=1)

# --- ANALYSIS 0: LEAGUE TABLE ---
teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
table = pd.DataFrame(index=teams, columns=['P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']).fillna(0)
for _, row in df.iterrows():
    h, a = row['HomeTeam'], row['AwayTeam']
    hg, ag, res = row['FTHG'], row['FTAG'], row['FTR']
    table.loc[h, ['P', 'GF', 'GA']] += [1, hg, ag]
    table.loc[a, ['P', 'GF', 'GA']] += [1, ag, hg]
    if res == 'H':
        table.loc[h, ['W', 'Pts']] += [1, 3]
        table.loc[a, 'L'] += 1
    elif res == 'A':
        table.loc[a, ['W', 'Pts']] += [1, 3]
        table.loc[h, 'L'] += 1
    else:
        table.loc[[h, a], ['D', 'Pts']] += [1, 1]
table['GD'] = table['GF'] - table['GA']
league_table = table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False)

# --- ANALYSIS 1: TEAM EFFICIENCY ---
home_eff = df.groupby('HomeTeam')[['FTHG', 'HST']].sum().rename(columns={'FTHG': 'Goals', 'HST': 'SOT'})
away_eff = df.groupby('AwayTeam')[['FTAG', 'AST']].sum().rename(columns={'FTAG': 'Goals', 'AST': 'SOT'})
efficiency = home_eff.add(away_eff, fill_value=0)
efficiency['Clinical_Rate'] = (efficiency['Goals'] / efficiency['SOT']).round(3)
top_efficient = efficiency.sort_values(by='Clinical_Rate', ascending=False)

# --- ANALYSIS 2: REFEREE DISCIPLINARY ---
df['TotalCards'] = df['HY'] + df['AY'] + df['HR'] + df['AR']
ref_stats = df.groupby('Referee').agg(Matches=('Date', 'count'), AvgCards=('TotalCards', 'mean'))
strict_refs = ref_stats[ref_stats['Matches'] >= 5].sort_values(by='AvgCards', ascending=False)

# --- ANALYSIS 3: HOME ADVANTAGE ---
outcomes = df['FTR'].value_counts(normalize=True) * 100

# --- ANALYSIS 4: COMEBACK KINGS ---
home_comebacks = df[(df['HTR'] == 'A') & (df['FTR'] == 'H')]['HomeTeam']
away_comebacks = df[(df['HTR'] == 'H') & (df['FTR'] == 'A')]['AwayTeam']
comeback_counts = pd.concat([home_comebacks, away_comebacks]).value_counts()

# --- ANALYSIS 5: BETTING ACCURACY ---
def get_fav(row):
    odds = {'H': row['B365H'], 'D': row['B365D'], 'A': row['B365A']}
    return min(odds, key=odds.get)
df['Favorite'] = df.apply(get_fav, axis=1)
fav_win_rate = (df['Favorite'] == df['FTR']).mean() * 100

# --- ANALYSIS 6: CLEAN SHEETS ---
home_cs = df[df['FTAG'] == 0]['HomeTeam'].value_counts()
away_cs = df[df['FTHG'] == 0]['AwayTeam'].value_counts()
clean_sheets = home_cs.add(away_cs, fill_value=0).sort_values(ascending=False)

# --- ANALYSIS 7: SECOND HALF SPECIALISTS ---
df['Home_1H'], df['Home_2H'] = df['HTHG'], df['FTHG'] - df['HTHG']
df['Away_1H'], df['Away_2H'] = df['HTAG'], df['FTAG'] - df['HTAG']
h_goals = df.groupby('HomeTeam')[['Home_1H', 'Home_2H']].sum().rename(columns={'Home_1H': 'Goals_1H', 'Home_2H': 'Goals_2H'})
a_goals = df.groupby('AwayTeam')[['Away_1H', 'Away_2H']].sum().rename(columns={'Away_1H': 'Goals_1H', 'Away_2H': 'Goals_2H'})
half_stats = h_goals.add(a_goals, fill_value=0)
half_stats['2H_Increase'] = half_stats['Goals_2H'] - half_stats['Goals_1H']
specialists = half_stats.sort_values(by='2H_Increase', ascending=False)

# --- ANALYSIS 8: CORNER CONVERSION ---
h_corn = df.groupby('HomeTeam')[['HC', 'FTHG']].sum().rename(columns={'HC': 'Corners', 'FTHG': 'Goals'})
a_corn = df.groupby('AwayTeam')[['AC', 'FTAG']].sum().rename(columns={'AC': 'Corners', 'FTAG': 'Goals'})
corn_total = h_corn.add(a_corn, fill_value=0)
corn_total['Goals_per_Corner'] = (corn_total['Goals'] / corn_total['Corners']).round(4)
top_corner_conv = corn_total.sort_values(by='Goals_per_Corner', ascending=False)

# --- ANALYSIS 9: AGGRESSION VS SUCCESS ---
h_agg = df.groupby('HomeTeam')[['HF', 'HomePts']].sum().rename(columns={'HF': 'Fouls', 'HomePts': 'Points'})
a_agg = df.groupby('AwayTeam')[['AF', 'AwayPts']].sum().rename(columns={'AF': 'Fouls', 'AwayPts': 'Points'})
aggression = h_agg.add(a_agg, fill_value=0)
corr_fouls_pts = aggression['Fouls'].corr(aggression['Points'])

# --- ANALYSIS 10: HOME VS AWAY SPLIT ---
ha_split = pd.DataFrame({'HomePts': df.groupby('HomeTeam')['HomePts'].sum(), 'AwayPts': df.groupby('AwayTeam')['AwayPts'].sum()})
ha_split['TotalPts'] = ha_split['HomePts'] + ha_split['AwayPts']
ha_split['Home_Contribution_%'] = (ha_split['HomePts'] / ha_split['TotalPts'] * 100).round(1)

# --- ANALYSIS 11: ENTERTAINMENT FACTOR ---
df['TotalGoals'] = df['FTHG'] + df['FTAG']
df['Over25'] = df['TotalGoals'] > 2.5
o25_stats = (df.groupby('HomeTeam')['Over25'].agg(['sum', 'count']).add(df.groupby('AwayTeam')['Over25'].agg(['sum', 'count']), fill_value=0))
o25_stats['Over25_Rate_%'] = (o25_stats['sum'] / o25_stats['count'] * 100).round(1)

# --- ANALYSIS 12: SHOT QUALITY ---
h_acc = df.groupby('HomeTeam')[['HST', 'HS']].sum()
a_acc = df.groupby('AwayTeam')[['AST', 'AS']].sum().rename(columns={'AST': 'HST', 'AS': 'HS'})
acc_total = h_acc.add(a_acc, fill_value=0)
acc_total['Accuracy_%'] = (acc_total['HST'] / acc_total['HS'] * 100).round(1)

# --- ANALYSIS 13: REPUTATION INDEX ---
h_rep = df.groupby('HomeTeam')[['HY', 'HF']].sum()
a_rep = df.groupby('AwayTeam')[['AY', 'AF']].sum().rename(columns={'AY': 'HY', 'AF': 'HF'})
rep_total = h_rep.add(a_rep, fill_value=0)
rep_total['CardsPer100Fouls'] = (rep_total['HY'] / rep_total['HF'] * 100).round(1)

# --- ANALYSIS 14: REFEREE BIAS ---
df['YC_Away_Bias'] = df['AY'] - df['HY']
ref_bias = df.groupby('Referee').agg(
    Matches=('Date', 'count'),
    HomeWinRate=('FTR', lambda x: (x == 'H').mean() * 100),
    AvgAwayCardBias=('YC_Away_Bias', 'mean')
).query('Matches >= 5').sort_values(by='HomeWinRate', ascending=False)

# --- ANALYSIS 15: BETTING ROI ---
bookie_map = {'B365': 'Bet365', 'BFD': 'Betfair', 'BMGM': 'BetMGM', 'BV': 'BetVictor', 
              'BW': 'Bwin', 'CL': 'Cloudbet', 'LB': 'Ladbrokes', 'PS': 'Pinnacle', 
              'Max': 'Market Max', 'Avg': 'Market Avg', 'BFE': 'Betfair Exchange'}

bookie_prefixes = sorted(list(set([c[:-1] for c in df.columns if c.endswith('H') and f'{c[:-1]}D' in df.columns and f'{c[:-1]}A' in df.columns])))

roi_data = []
for p in bookie_prefixes:
    def calc_roi(row, prefix, bet_type):
        try:
            odds = {'H': row[f'{prefix}H'], 'D': row[f'{prefix}D'], 'A': row[f'{prefix}A']}
            if any(pd.isna(v) for v in odds.values()): return np.nan
            pick = min(odds, key=odds.get) if bet_type == 'fav' else max(odds, key=odds.get)
            return (odds[pick] - 1) if pick == row['FTR'] else -1
        except: return np.nan

    f_roi = df.apply(lambda r: calc_roi(r, p, 'fav'), axis=1).mean() * 100
    d_roi = df.apply(lambda r: calc_roi(r, p, 'dog'), axis=1).mean() * 100
    
    is_closing = p.endswith('C')
    base_code = p[:-1] if is_closing else p
    name = bookie_map.get(base_code, base_code)
    label = f"{name} (Closing)" if is_closing else f"{name} (Opening)"
    
    roi_data.append({'Bookmaker': label, 'Fav_ROI': round(f_roi, 2), 'Underdog_ROI': round(d_roi, 2)})

roi_table = pd.DataFrame(roi_data)

# --- ANALYSIS 16: DEFENSIVE WALL ---
h_def = df.groupby('HomeTeam')[['AST', 'FTAG']].sum().rename(columns={'AST': 'SOT_Conc', 'FTAG': 'G_Conc'})
a_def = df.groupby('AwayTeam')[['HST', 'FTHG']].sum().rename(columns={'HST': 'SOT_Conc', 'FTHG': 'G_Conc'})
def_wall = h_def.add(a_def, fill_value=0)
def_wall['SOT_per_Goal'] = (def_wall['SOT_Conc'] / def_wall['G_Conc']).round(2)

# --- ANALYSIS 17: RED CARD CONSEQUENCE ---
def get_red_card_drilldown(df):
    red_card_events = []
    h_reds = df[df['HR'] > 0]
    for _, row in h_reds.iterrows():
        outcome = 'Win' if row['FTR'] == 'H' else ('Draw' if row['FTR'] == 'D' else 'Loss')
        red_card_events.append({'Team': row['HomeTeam'], 'Outcome': outcome})
    
    a_reds = df[df['AR'] > 0]
    for _, row in a_reds.iterrows():
        outcome = 'Win' if row['FTR'] == 'A' else ('Draw' if row['FTR'] == 'D' else 'Loss')
        red_card_events.append({'Team': row['AwayTeam'], 'Outcome': outcome})
    
    if not red_card_events:
        return pd.DataFrame(columns=['Total', 'Win%', 'Draw%', 'Loss%'])
    
    rc_df = pd.DataFrame(red_card_events)
    drilldown = rc_df.groupby('Team')['Outcome'].value_counts().unstack(fill_value=0)
    
    for col in ['Win', 'Draw', 'Loss']:
        if col not in drilldown.columns:
            drilldown[col] = 0
    
    drilldown['Total'] = drilldown[['Win', 'Draw', 'Loss']].sum(axis=1)
    drilldown['Win%'] = (drilldown['Win'] / drilldown['Total'] * 100).round(1)
    drilldown['Draw%'] = (drilldown['Draw'] / drilldown['Total'] * 100).round(1)
    drilldown['Loss%'] = (drilldown['Loss'] / drilldown['Total'] * 100).round(1)
    
    return drilldown[['Total', 'Win', 'Draw', 'Loss', 'Win%', 'Draw%', 'Loss%']].sort_values(by=['Total', 'Win%'], ascending=False)

red_card_report = get_red_card_drilldown(df)

# --- ANALYSIS 18: QUICK STATS ---
quick_stats = {
    'home_win_pct': round((df['FTR']=='H').mean()*100, 1),
    'avg_goals_per_game': round(df['TotalGoals'].mean(), 2),
    'most_common_referee': df['Referee'].value_counts().idxmax()
}

# --- BUILD JSON OUTPUT ---
output = {
    'last_updated': datetime.now().isoformat(),
    'analysis': {
        '0_league_table': league_table.reset_index().rename(columns={'index': 'Team'}).to_dict('records'),
        '1_team_efficiency': top_efficient.reset_index().rename(columns={'index': 'Team'}).to_dict('records'),
        '2_referee_discipline': strict_refs.reset_index().rename(columns={'index': 'Referee'}).to_dict('records'),
        '3_home_advantage': {
            'home_win_pct': round(outcomes.get('H', 0), 1),
            'away_win_pct': round(outcomes.get('A', 0), 1),
            'draw_pct': round(outcomes.get('D', 0), 1)
        },
        '4_comeback_kings': comeback_counts.reset_index().rename(columns={'index': 'Team', 0: 'Comebacks'}).to_dict('records'),
        '5_betting_accuracy': {
            'favorite_win_rate': round(fav_win_rate, 2)
        },
        '6_clean_sheets': clean_sheets.reset_index().rename(columns={'index': 'Team', 0: 'Clean_Sheets'}).to_dict('records'),
        '7_second_half_specialists': specialists.reset_index().rename(columns={'index': 'Team'}).to_dict('records'),
        '8_corner_conversion': top_corner_conv.reset_index().rename(columns={'index': 'Team'}).to_dict('records'),
        '9_aggression_vs_success': {
            'correlation': round(corr_fouls_pts, 3),
            'data': aggression.reset_index().rename(columns={'index': 'Team'}).sort_values(by='Points', ascending=False).to_dict('records')
        },
        '10_home_away_split': ha_split.reset_index().rename(columns={'index': 'Team'}).sort_values(by='Home_Contribution_%', ascending=False).to_dict('records'),
        '11_entertainment_factor': o25_stats.reset_index().rename(columns={'index': 'Team'}).sort_values(by='Over25_Rate_%', ascending=False).to_dict('records'),
        '12_shot_quality': acc_total.reset_index().rename(columns={'index': 'Team'}).sort_values(by='Accuracy_%', ascending=False).to_dict('records'),
        '13_reputation_index': rep_total.reset_index().rename(columns={'index': 'Team'}).sort_values(by='CardsPer100Fouls', ascending=False).to_dict('records'),
        '14_referee_bias': ref_bias.reset_index().rename(columns={'index': 'Referee'}).head(10).to_dict('records'),
        '15_betting_roi': roi_table.to_dict('records'),
        '16_defensive_wall': def_wall.reset_index().rename(columns={'index': 'Team'}).sort_values(by='SOT_per_Goal', ascending=False).to_dict('records'),
        '17_red_card_survival': red_card_report.reset_index().rename(columns={'index': 'Team'}).to_dict('records') if not red_card_report.empty else [],
        '18_quick_stats': quick_stats
    }
}

# Save to JSON file
with open('football_data.json', 'w') as f:
    json.dump(output, f, indent=2)

print("✅ Data processed successfully!")
print(f"📊 Last updated: {output['last_updated']}")
print(f"📈 Total matches analyzed: {len(df)}")
