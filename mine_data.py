# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:53:50 2021

@author: Kevin Sebineza
"""

import sd_algorithm
import pandas as pd

sd = sd_algorithm.SDAlgorithm()

# Function to get news content from url
def get_content(url):
    sd.url = url
    text = sd.analyze_page()
    return text

"""
The following are the online newspapers that I'm going to mine contents from
1. BBC
2. The Newtimes
3. The guardian
4. New york times

"""

""" Getting contents from BBC """


# Political news
sd = sd_algorithm.SDAlgorithm()
bbcPol1 = get_content('https://www.bbc.com/news/uk-56572775')
sd = sd_algorithm.SDAlgorithm()
bbcPol2 = get_content('https://www.bbc.com/news/world-latin-america-56581131')
sd = sd_algorithm.SDAlgorithm()
bbcPol3 = get_content('https://www.bbc.com/news/world-asia-39595989')
sd = sd_algorithm.SDAlgorithm()
bbcPol4 = get_content('https://www.bbc.com/news/technology-55569604')
sd = sd_algorithm.SDAlgorithm()
bbcPol5 = get_content('https://www.bbc.com/news/uk-northern-ireland-56566468')

# Business news
sd = sd_algorithm.SDAlgorithm()
bbcBus1 = get_content('https://www.bbc.com/news/business-56581614')
sd = sd_algorithm.SDAlgorithm()
bbcBus2 = get_content('https://www.bbc.com/news/business-56441829')
sd = sd_algorithm.SDAlgorithm()
bbcBus3 = get_content('https://www.bbc.com/news/business-56578445')
sd = sd_algorithm.SDAlgorithm()
bbcBus4 = get_content('https://www.bbc.com/news/technology-56154543')
sd = sd_algorithm.SDAlgorithm()
bbcBus5 = get_content('https://www.bbc.com/news/business-56585697')

# Sport news
sd = sd_algorithm.SDAlgorithm()
bbcSpo1 = get_content('https://www.bbc.com/sport/football/56505758')
sd = sd_algorithm.SDAlgorithm()
bbcSpo2 = get_content('https://www.bbc.com/sport/football/56586894')
sd = sd_algorithm.SDAlgorithm()
bbcSpo3 = get_content('https://www.bbc.com/sport/golf/56559194')
sd = sd_algorithm.SDAlgorithm()
bbcSpo4 = get_content('https://www.bbc.com/sport/football/56584938')
sd = sd_algorithm.SDAlgorithm()
bbcSpo5 = get_content('https://www.bbc.com/sport/formula1/56557657')

# Entertainment
sd = sd_algorithm.SDAlgorithm()
bbcEnt1 = get_content('https://www.bbc.com/culture/article/20210315-the-mystery-of-lost-rock-genius-lee-mavers')
sd = sd_algorithm.SDAlgorithm()
bbcEnt2 = get_content('https://www.bbc.com/culture/article/20210219-why-pop-stars-are-having-prosthetic-makeovers')
sd = sd_algorithm.SDAlgorithm()
bbcEnt3 = get_content('https://www.bbc.com/culture/article/20210216-the-forgotten-story-of-americas-first-black-superstars')
sd = sd_algorithm.SDAlgorithm()
bbcEnt4 = get_content('https://www.bbc.com/culture/article/20200923-is-princes-sign-o-the-times-the-greatest-album-of-all-time')
sd = sd_algorithm.SDAlgorithm()
bbcEnt5 = get_content('https://www.bbc.com/culture/article/20140924-was-hendrix-really-that-original')

""" Getting contents from The newtimes """

# Political news
sd = sd_algorithm.SDAlgorithm()
newPol1 = get_content('https://www.newtimes.co.rw/news/premier-ngirente-attends-car-president-touaderas-swearing-ceremony')
sd = sd_algorithm.SDAlgorithm()
newPol2 = get_content('https://www.newtimes.co.rw/news/nine-takeaways-tuesdays-covid-19-news-conference')
sd = sd_algorithm.SDAlgorithm()
newPol3 = get_content('https://www.newtimes.co.rw/opinions/rusesabaginas-arrest-international-law')
sd = sd_algorithm.SDAlgorithm()
newPol4 = get_content('https://www.newtimes.co.rw/news/kagame-stresses-need-modernizing-international-debt-architecture')
sd = sd_algorithm.SDAlgorithm()
newPol5 = get_content('https://www.newtimes.co.rw/international/japan-appoints-minister-loneliness-tackle-covid-induced-suicides')

# Business news
sd = sd_algorithm.SDAlgorithm()
newBus1 = get_content('https://www.newtimes.co.rw/news/what-next-after-shareholders-agree-dissolve-crystal-telecom')
sd = sd_algorithm.SDAlgorithm()
newBus2 = get_content('https://www.newtimes.co.rw/news/going-green-volkswagen-rwanda-unveils-new-motor-charging-station')
sd = sd_algorithm.SDAlgorithm()
newBus3 = get_content('https://www.newtimes.co.rw/news/featured-bk-group-registered-rwf384bn-after-tax-profit-2020')
sd = sd_algorithm.SDAlgorithm()
newBus4 = get_content('https://www.newtimes.co.rw/news/rwanda-banks-tech-get-18-households-out-food-insecurity')
sd = sd_algorithm.SDAlgorithm()
newBus5 = get_content('https://www.newtimes.co.rw/opinions/editorial-increased-consumption-local-products-could-drive-recovery')

# Sport news
sd = sd_algorithm.SDAlgorithm()
newSpo1 = get_content('https://www.newtimes.co.rw/sports/uci-assess-rwandas-bid-host-2025-road-cycling-championship')
sd = sd_algorithm.SDAlgorithm()
newSpo2 = get_content('https://www.newtimes.co.rw/sports/basketball-nba-sets-date-inaugural-bal-tourney')
sd = sd_algorithm.SDAlgorithm()
newSpo3 = get_content('https://www.newtimes.co.rw/sports/hertier-irasubiza-young-talent-dreams-becoming-big-star-local-football')
sd = sd_algorithm.SDAlgorithm()
newSpo4 = get_content('https://www.newtimes.co.rw/sports/mugisha-could-miss-tour-du-rwanda-2021')
sd = sd_algorithm.SDAlgorithm()
newSpo5 = get_content('https://www.newtimes.co.rw/sports/tasks-await-mukura-head-coach-rodolfo-zapata')

# Entertainment
sd = sd_algorithm.SDAlgorithm()
newEnt1 = get_content('https://www.newtimes.co.rw/entertainment/miss-talent-2021-umutoniwase-named-hdi-youth-ambassador')
sd = sd_algorithm.SDAlgorithm()
newEnt2 = get_content('https://www.newtimes.co.rw/entertainment/close-stand-comedian-joshua')
sd = sd_algorithm.SDAlgorithm()
newEnt3 = get_content('https://www.newtimes.co.rw/entertainment/trees-peace-movie-premiere-next-month')
sd = sd_algorithm.SDAlgorithm()
newEnt4 = get_content('https://www.newtimes.co.rw/entertainment/celebrating-womens-creativity-artists-live-stream-concert')
sd = sd_algorithm.SDAlgorithm()
newEnt5 = get_content('https://www.newtimes.co.rw/entertainment/grace-ingabire-crowned-miss-rwanda-2021')

""" Getting contents from The guardian """

# Political news
sd = sd_algorithm.SDAlgorithm()
guaPol1 = get_content('https://www.theguardian.com/world/2021/mar/30/who-criticises-chinas-data-sharing-as-it-releases-covid-origins-report')
sd = sd_algorithm.SDAlgorithm()
guaPol2 = get_content('https://www.youtube.com/watch?v=ve_0h4Y8nuI&list=PLhTjy8cBISEqkN-5Ku_kXG4QW33sxQo0t')
sd = sd_algorithm.SDAlgorithm()
guaPol3 = get_content('https://www.theguardian.com/uk-news/2021/mar/30/report-into-policing-of-sarah-everard-vigil-a-missed-opportunity')
sd = sd_algorithm.SDAlgorithm()
guaPol4 = get_content('https://www.theguardian.com/education/2021/mar/30/ofsted-chief-asked-for-greater-powers-to-monitor-private-schools')
sd = sd_algorithm.SDAlgorithm()
guaPol5 = get_content('https://www.theguardian.com/global-development/2021/mar/31/in-the-middle-of-a-war-zone-thousands-flee-as-venezuela-troops-and-colombia-rebels-clash')

# Business news
sd = sd_algorithm.SDAlgorithm()
guaBus1 = get_content('https://www.theguardian.com/business/2021/mar/30/infrastructure-projects-should-use-more-uk-steel-says-trade-body')
sd = sd_algorithm.SDAlgorithm()
guaBus2 = get_content('https://www.theguardian.com/business/2021/mar/31/fullers-to-raise-cash-after-burning-through-up-to-5m-a-month-covid')
sd = sd_algorithm.SDAlgorithm()
guaBus3 = get_content('https://www.theguardian.com/business/2021/mar/31/uk-economy-savings-expansion-recovery')
sd = sd_algorithm.SDAlgorithm()
guaBus4 = get_content('https://www.theguardian.com/commentisfree/2021/mar/30/biden-tariffs-brexit-britain-eu-big-tech')
sd = sd_algorithm.SDAlgorithm()
guaBus5 = get_content('https://www.theguardian.com/money/2021/mar/29/britons-pay-back-most-on-debt-in-27-years-as-credit-card-spending-slumps-covid')

# Sport news
sd = sd_algorithm.SDAlgorithm()
guaSpo1 = get_content('https://www.theguardian.com/football/2021/mar/30/wales-czech-republic-world-cup-qualifier-match-report')
sd = sd_algorithm.SDAlgorithm()
guaSpo2 = get_content('https://www.theguardian.com/football/2021/mar/31/agents-ready-for-war-with-fifa-over-new-rules-raiola-barnett-football-forum')
sd = sd_algorithm.SDAlgorithm()
guaSpo3 = get_content('https://www.theguardian.com/football/2021/mar/31/trevoh-chalobah-lorient-chelsea-psg-mark-neymar')
sd = sd_algorithm.SDAlgorithm()
guaSpo4 = get_content('https://www.theguardian.com/football/2021/mar/30/fa-to-use-psychological-profiling-to-help-appoint-england-women-captain')
sd = sd_algorithm.SDAlgorithm()
guaSpo5 = get_content('https://www.theguardian.com/sport/2021/mar/24/british-masters-golf-bid-for-crowds-at-proposed-covid-pilot-event-in-may')

# Entertainment
sd = sd_algorithm.SDAlgorithm()
guaEnt1 = get_content('https://www.theguardian.com/music/2021/mar/30/cardi-b-haircare-products-afro-latina-hair')
sd = sd_algorithm.SDAlgorithm()
guaEnt2 = get_content('https://www.theguardian.com/music/2021/mar/31/britney-spears-responds-to-documentary-about-her-life-framing')
sd = sd_algorithm.SDAlgorithm()
guaEnt3 = get_content('https://www.theguardian.com/music/2021/mar/30/antonio-pappano-to-replace-simon-rattle-at-london-symphony-orchestra')
sd = sd_algorithm.SDAlgorithm()
guaEnt4 = get_content('https://www.theguardian.com/music/2021/mar/30/lil-nas-x-montero-call-me-by-your-name-twitter')
sd = sd_algorithm.SDAlgorithm()
guaEnt5 = get_content('https://www.theguardian.com/culture/2021/mar/31/glastonbury-live-stream-festival-coldplay-michael-kiwanuka-and-haim-to-perform')

""" Getting contents from NewYork times """

# Political news
sd = sd_algorithm.SDAlgorithm()
neyPol1 = get_content('https://www.nytimes.com/2021/03/29/us/politics/biden-virus-vaccine.html')
sd = sd_algorithm.SDAlgorithm()
neyPol2 = get_content('https://www.nytimes.com/2021/03/30/us/politics/democrats-voting-rights-bill.html')
sd = sd_algorithm.SDAlgorithm()
neyPol3 = get_content('https://www.nytimes.com/2021/03/31/us/politics/capitol-police-lawsuit-trump.html')
sd = sd_algorithm.SDAlgorithm()
neyPol4 = get_content('https://www.nytimes.com/2021/03/31/us/supreme-court-ncaa.html')
sd = sd_algorithm.SDAlgorithm()
neyPol5 = get_content('https://www.nytimes.com/2021/03/30/us/politics/blinken-human-rights-women.html')

# Business news
sd = sd_algorithm.SDAlgorithm()
neyBus1 = get_content('https://www.nytimes.com/2021/03/29/business/archegos-hwang-viacomcbs-discovery.html')
sd = sd_algorithm.SDAlgorithm()
neyBus2 = get_content('https://www.nytimes.com/2021/03/30/business/ffel-student-loans-paused.html')
sd = sd_algorithm.SDAlgorithm()
neyBus3 = get_content('https://www.nytimes.com/2021/03/27/at-home/how-to-save-at-the-grocery-store.html')
sd = sd_algorithm.SDAlgorithm()
neyBus4 = get_content('https://www.nytimes.com/2021/03/20/style/spending-rich-people.html')
sd = sd_algorithm.SDAlgorithm()
neyBus5 = get_content('https://www.nytimes.com/2021/03/12/your-money/taxes/2020-taxes-work-from-home.html')

# Sport news
sd = sd_algorithm.SDAlgorithm()
neySpo1 = get_content('https://www.nytimes.com/2021/03/26/sports/soccer/jesse-lingard-england-west-ham.html')
sd = sd_algorithm.SDAlgorithm()
neySpo2 = get_content('https://www.nytimes.com/2021/03/30/sports/ncaabasketball/march-madness-final-four-gonzaga.html')
sd = sd_algorithm.SDAlgorithm()
neySpo3 = get_content('https://www.nytimes.com/2021/03/31/sports/ncaabasketball/ncaa-womens-final-four-preview.html')
sd = sd_algorithm.SDAlgorithm()
neySpo4 = get_content('https://www.nytimes.com/2021/03/31/sports/soccer/soccer-streetwear-juventus.html')
sd = sd_algorithm.SDAlgorithm()
neySpo5 = get_content('https://www.nytimes.com/interactive/2021/03/29/sports/soccer/champions-league-new-format.html')

urls = ['https://www.bbc.com/news/uk-56572775', 'https://www.bbc.com/news/world-latin-america-56581131', 'tps://www.bbc.com/news/world-asia-39595989', 'https://www.bbc.com/news/technology-55569604','https://www.bbc.com/news/uk-northern-ireland-56566468',
        'https://www.bbc.com/news/business-56581614','https://www.bbc.com/news/business-56441829','https://www.bbc.com/news/business-56578445','https://www.bbc.com/news/technology-56154543','https://www.bbc.com/news/business-56585697',
        'https://www.bbc.com/sport/football/56505758','https://www.bbc.com/sport/football/56586894','https://www.bbc.com/sport/golf/56559194','https://www.bbc.com/sport/football/56584938','https://www.bbc.com/sport/formula1/56557657',
        'https://www.bbc.com/culture/article/20210315-the-mystery-of-lost-rock-genius-lee-mavers','https://www.bbc.com/culture/article/20210219-why-pop-stars-are-having-prosthetic-makeovers','https://www.bbc.com/culture/article/20210216-the-forgotten-story-of-americas-first-black-superstars','https://www.bbc.com/culture/article/20200923-is-princes-sign-o-the-times-the-greatest-album-of-all-time','https://www.bbc.com/culture/article/20140924-was-hendrix-really-that-original',
        'https://www.newtimes.co.rw/news/premier-ngirente-attends-car-president-touaderas-swearing-ceremony','https://www.newtimes.co.rw/news/nine-takeaways-tuesdays-covid-19-news-conference','https://www.newtimes.co.rw/opinions/rusesabaginas-arrest-international-law','https://www.newtimes.co.rw/news/kagame-stresses-need-modernizing-international-debt-architecture','https://www.newtimes.co.rw/international/japan-appoints-minister-loneliness-tackle-covid-induced-suicides',
        'https://www.newtimes.co.rw/news/what-next-after-shareholders-agree-dissolve-crystal-telecom','https://www.newtimes.co.rw/news/going-green-volkswagen-rwanda-unveils-new-motor-charging-station','https://www.newtimes.co.rw/news/featured-bk-group-registered-rwf384bn-after-tax-profit-2020','https://www.newtimes.co.rw/news/rwanda-banks-tech-get-18-households-out-food-insecurity','https://www.newtimes.co.rw/opinions/editorial-increased-consumption-local-products-could-drive-recovery',
        'https://www.newtimes.co.rw/sports/uci-assess-rwandas-bid-host-2025-road-cycling-championship','https://www.newtimes.co.rw/sports/basketball-nba-sets-date-inaugural-bal-tourney','https://www.newtimes.co.rw/sports/hertier-irasubiza-young-talent-dreams-becoming-big-star-local-football','https://www.newtimes.co.rw/sports/mugisha-could-miss-tour-du-rwanda-2021','https://www.newtimes.co.rw/sports/tasks-await-mukura-head-coach-rodolfo-zapata',
        'https://www.newtimes.co.rw/entertainment/miss-talent-2021-umutoniwase-named-hdi-youth-ambassador','https://www.newtimes.co.rw/entertainment/close-stand-comedian-joshua','https://www.newtimes.co.rw/entertainment/trees-peace-movie-premiere-next-month','https://www.newtimes.co.rw/entertainment/celebrating-womens-creativity-artists-live-stream-concert','https://www.newtimes.co.rw/entertainment/grace-ingabire-crowned-miss-rwanda-2021',
        'https://www.theguardian.com/world/2021/mar/30/who-criticises-chinas-data-sharing-as-it-releases-covid-origins-report','https://www.youtube.com/watch?v=ve_0h4Y8nuI&list=PLhTjy8cBISEqkN-5Ku_kXG4QW33sxQo0t','https://www.theguardian.com/uk-news/2021/mar/30/report-into-policing-of-sarah-everard-vigil-a-missed-opportunity','https://www.theguardian.com/education/2021/mar/30/ofsted-chief-asked-for-greater-powers-to-monitor-private-schools','https://www.theguardian.com/global-development/2021/mar/31/in-the-middle-of-a-war-zone-thousands-flee-as-venezuela-troops-and-colombia-rebels-clash',
        'https://www.theguardian.com/business/2021/mar/30/infrastructure-projects-should-use-more-uk-steel-says-trade-body','https://www.theguardian.com/business/2021/mar/31/fullers-to-raise-cash-after-burning-through-up-to-5m-a-month-covid','https://www.theguardian.com/business/2021/mar/31/uk-economy-savings-expansion-recovery','https://www.theguardian.com/commentisfree/2021/mar/30/biden-tariffs-brexit-britain-eu-big-tech','https://www.theguardian.com/money/2021/mar/29/britons-pay-back-most-on-debt-in-27-years-as-credit-card-spending-slumps-covid',
        'https://www.theguardian.com/football/2021/mar/30/wales-czech-republic-world-cup-qualifier-match-report','https://www.theguardian.com/football/2021/mar/31/agents-ready-for-war-with-fifa-over-new-rules-raiola-barnett-football-forum','https://www.theguardian.com/football/2021/mar/31/trevoh-chalobah-lorient-chelsea-psg-mark-neymar','https://www.theguardian.com/football/2021/mar/30/fa-to-use-psychological-profiling-to-help-appoint-england-women-captain','https://www.theguardian.com/sport/2021/mar/24/british-masters-golf-bid-for-crowds-at-proposed-covid-pilot-event-in-may',
        'https://www.theguardian.com/music/2021/mar/30/cardi-b-haircare-products-afro-latina-hair','https://www.theguardian.com/music/2021/mar/31/britney-spears-responds-to-documentary-about-her-life-framing','https://www.theguardian.com/music/2021/mar/30/antonio-pappano-to-replace-simon-rattle-at-london-symphony-orchestra','https://www.theguardian.com/music/2021/mar/30/lil-nas-x-montero-call-me-by-your-name-twitter','https://www.theguardian.com/culture/2021/mar/31/glastonbury-live-stream-festival-coldplay-michael-kiwanuka-and-haim-to-perform',
        'https://www.nytimes.com/2021/03/29/us/politics/biden-virus-vaccine.html','https://www.nytimes.com/2021/03/30/us/politics/democrats-voting-rights-bill.html','https://www.nytimes.com/2021/03/31/us/politics/capitol-police-lawsuit-trump.html','https://www.nytimes.com/2021/03/31/us/supreme-court-ncaa.html','https://www.nytimes.com/2021/03/30/us/politics/blinken-human-rights-women.html',
        'https://www.nytimes.com/2021/03/29/business/archegos-hwang-viacomcbs-discovery.html','https://www.nytimes.com/2021/03/30/business/ffel-student-loans-paused.html','https://www.nytimes.com/2021/03/27/at-home/how-to-save-at-the-grocery-store.html','https://www.nytimes.com/2021/03/20/style/spending-rich-people.html','https://www.nytimes.com/2021/03/12/your-money/taxes/2020-taxes-work-from-home.html',
        'https://www.nytimes.com/2021/03/26/sports/soccer/jesse-lingard-england-west-ham.html','https://www.nytimes.com/2021/03/30/sports/ncaabasketball/march-madness-final-four-gonzaga.html','https://www.nytimes.com/2021/03/31/sports/ncaabasketball/ncaa-womens-final-four-preview.html','https://www.nytimes.com/2021/03/31/sports/soccer/soccer-streetwear-juventus.html','https://www.nytimes.com/interactive/2021/03/29/sports/soccer/champions-league-new-format.html',
        'https://www.nytimes.com/2021/03/29/arts/music/sasha-velour-drag-opera.html','https://www.nytimes.com/2021/03/29/style/arizona-marijuana-legalization.html','https://www.nytimes.com/2021/03/30/arts/music/freddie-redd-dead.html','https://www.nytimes.com/2021/03/29/arts/music/justin-bieber-justice-billboard-chart.html','https://www.nytimes.com/2021/03/28/arts/music/biggest-indoor-rock-concert-barcelona.html']
# Entertainment
sd = sd_algorithm.SDAlgorithm()
neyEnt1 = get_content('https://www.nytimes.com/2021/03/29/arts/music/sasha-velour-drag-opera.html')
sd = sd_algorithm.SDAlgorithm()
neyEnt2 = get_content('https://www.nytimes.com/2021/03/29/style/arizona-marijuana-legalization.html')
sd = sd_algorithm.SDAlgorithm()
neyEnt3 = get_content('https://www.nytimes.com/2021/03/30/arts/music/freddie-redd-dead.html')
sd = sd_algorithm.SDAlgorithm()
neyEnt4 = get_content('https://www.nytimes.com/2021/03/29/arts/music/justin-bieber-justice-billboard-chart.html')
sd = sd_algorithm.SDAlgorithm()
neyEnt5 = get_content('https://www.nytimes.com/2021/03/28/arts/music/biggest-indoor-rock-concert-barcelona.html')

""" Getting data into a dataframe """
labels = ['politics', 'politics', 'politics', 'politics', 'politics', 'business', 'business', 'business', 'business', 'business', 
          'sports', 'sports', 'sports', 'sports', 'sports', 'entertainment', 'entertainment', 'entertainment', 'entertainment', 'entertainment',
          'politics', 'politics', 'politics', 'politics', 'politics', 'business', 'business', 'business', 'business', 'business', 
          'sports', 'sports', 'sports', 'sports', 'sports', 'entertainment', 'entertainment', 'entertainment', 'entertainment', 'entertainment',
          'politics', 'politics', 'politics', 'politics', 'politics', 'business', 'business', 'business', 'business', 'business', 
          'sports', 'sports', 'sports', 'sports', 'sports', 'entertainment', 'entertainment', 'entertainment', 'entertainment', 'entertainment',
          'politics', 'politics', 'politics', 'politics', 'politics', 'business', 'business', 'business', 'business', 'business', 
          'sports', 'sports', 'sports', 'sports', 'sports', 'entertainment', 'entertainment', 'entertainment', 'entertainment', 'entertainment']



dictionary = {"article":[bbcPol1, bbcPol2, bbcPol3, bbcPol4, bbcPol5, bbcBus1, bbcBus2, bbcBus3, bbcBus4, bbcBus5, 
                         bbcSpo1, bbcSpo2, bbcSpo3, bbcSpo4, bbcSpo5, bbcEnt1, bbcEnt2, bbcEnt3, bbcEnt4, bbcEnt5, 
                         newPol1, newPol2, newPol3, newPol4, newPol5, newBus1, newBus2, newBus3, newBus4, newBus5, 
                         newSpo1, newSpo2, newSpo3, newSpo4, newSpo5, newEnt1, newSpo2, newSpo3, newSpo4, newSpo5, 
                         guaPol1, guaPol2, guaPol3, guaPol4, guaPol5, guaBus1, guaBus2, guaBus3, guaBus4, guaBus5, 
                         guaSpo1, guaSpo2, guaSpo3, guaSpo4, guaSpo5, guaEnt1, guaEnt2, guaEnt3, guaEnt4, guaEnt5, 
                         neyPol1, neyPol2, neyPol3, neyPol4, neyPol5, neyBus1, neyBus2, neyBus3, neyBus4, neyBus5, 
                         neySpo1, neySpo2, neySpo3, neySpo4, neySpo5, neyEnt1, neyEnt2, neyEnt3, neyEnt4, neyEnt5, ], "category": labels, "url": urls}

df = pd.DataFrame(dictionary)
df.to_csv("mined_news_content.csv")
