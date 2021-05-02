

#financial_event['event_name'] = list of words associated with the event {word1,...,wordN}


FINANCIAL_EVENTS = {
    'product_launch':['product','new','launch','release','publish','present','unveil','upgrade','announce','reveal','invent','push','introduce','sponsor','plan','create','provide','unseal','expand','fresh','broaden','extend','submit','issue','highlight','address','discuss','fabricate','distribute','redistribute','reform','celebrate','outline','approve','propose','boost','overhaul','deliver','relaunch','allocate','bullish','upbeat','outperform','smash','leaderboard','upswing','buoyant','oversell'],
    'product_recall': ['recall','downgrade','stifel', 'slip', 'plunge', 'crush', 'overvalue', 'lower', 'bearish', 'fall', 'tumble', 'slump', 'nosedive', 'underperform', 'disappoint', 'downtrend', 'downswing'],
    'merge_acquisition': ['acquire','merge','merger','purchase', 'sale','takeover','reacquire', 'buy', 'partner', 'venture', 'sell',
'jointventure', 'expand', 'offer', 'buyout', 'megamerger','acquisition'],
    'financial': ['capital','earning','revenue', 'dividend','asset', 'stockholder','stakeholder','stake', 'stock', 'money', 'loan', 'equity','bond','quarter','result','profit','firstquarter','secondquarter','thirdquarter','junequarter','decemberquarter','share','forecast','investor','income','expectation','cost','topline','analystsexpectation','profitability','gross','payout','yielding','payable','retiree','buyback','earner','eps','etn','ric','tss','interest','diversify','exdividend','currency','fund','domesticstock','market','inflow','corporatebond','riskyasset','bourse','yield' ],'price_change': ['price','rise', 'surge','gain', 'slump','drop', 'fall','shrink','plunge','decline','stock','share', 'jump', 'climb', 'slip', 'dip', 'decline', 'soar', 'edge', 'slide', 'increase', 'decrease', 'weak', 'tumble', 'steady', 'buoy', 'sink', 'grow', 'flat', 'boost', 'uptick', 'bounce', 'push', 'rally','rebound', 'plummet', 'slowdown', 'growth', 'selloff', 'hit', 'contraction', 'slow', 'decelerate', 'deceleration', 'pressured', 'dive', 'downbeat', 'pullback', 'downturn', 'lowest', 'lower', 'steep', 'stabilise', 'worsen', 'gloom', 'skid', 'crimp', 'evaporate', 'recordlow', 'spike', 'weakest', 'weaken', 'drag', 'overdone', 'subdue', 'uptick', 'ease', 'wane' ],
    'legal':['law','suit','antitrust','fiscal','report','contract','tax','debt','judiciary','audit','legislation','ordinance','amendment','bill', 'prohibite','statute','court','lawmaker','judge','regulation','policy','constitution','criminalize','federal','illegal','ruling','legislature', 'senate','attorney','authority','abolish','prohibition','prosecutor','outlaw','tribunal','lawsuit','complaint','sue','infringement','dismissal','violation','motion','file','classaction','petition','litigation','disqualify','penalty'],
    'bankruptcy': ['bankruptcy','bankrupt']   
}

FINANCIAL_EVENTS_WEIGHTS = {'product_launch':3, 'product_recall':10, 'merge_acquisition':5,  'financial':2, 'price_change':5, 'legal':5, 'bankruptcy':10}

