# 1 Feedback Competition Drives Emotional 2 Bubbles in Information Ecosystems During 3 Health Crises

4 Juan Li 1, Jinlin Wu 2\*, Zhihang Liu 3\* 5 6 'School of Journalism and Communication, Lanzhou University, 730000 Lanzhou, 7 China 8 2School of Journalism and Communication, Lanzhou University, 730000 Lanzhou, 9 China 10 3Institute of Space and Earth Information Science, The Chinese University of Hong 11 Kong, Shatin, Hong Kong, China

13 Corresponding Authors: 14 Jinlin Wu 15 Lanzhou University 16 Tel: (+86)13367073600 17 Email: 220220941091@lzu.edu.cn 18 19 Zhihang Liu 20 The Chinese University of Hong Kong, Shatin 21 Tel: (+86)13143009355 22 Email: zhihangliu@cuhk.edu.hk

# 23 Classification:

24 Major: Physical Sciences; Social Sciences 25 Minor: Applied Physical Sciences; Social Sciences 26 Keywords: Emotional bubbles, Mixed- feedback mechanisms, Phase transitions, Information 27 ecosystems

28

29

30

# 31 Abstract

31 Abstract32 The COVID- 19 pandemic triggered widespread psychological distress and information chaos, 33 creating complex emotional dynamics that significantly disrupted public health responses. 34 However, it remains unclear how psychological sensitivity and information ecosystem 35 composition interact to drive these nonlinear emotional dynamics. Using sentiment analysis 36 of 294,434 Weibo posts during COVID- 19 and developing a mixed- feedback theoretical 37 framework, we reveal that emotional bubbles emerge from competing feedback mechanisms: 38 mainstream media provides stabilizing negative feedback while We media creates amplifying 39 positive feedback loops. We show that as individual psychological thresholds converge, 40 selective exposure behaviors systematically erode moderate emotional buffers, creating 41 critical transitions from continuous resilient dynamics to discontinuous fragile responses. 42 Phase transition analysis demonstrates that systems undergo qualitatively different behaviors 43 across psychological threshold space, with critical points determined by the balance between 44 stabilizing and amplifying feedback mechanisms. Network simulations validate that 45 progressive removal of mainstream media triggers cascading instability, transforming resilient 46 emotional ecosystems into brittle configurations prone to sudden collapse. Our findings reveal 47 that maintaining balanced feedback between stabilizing and amplifying information sources is 48 crucial for preventing sudden emotional collapses during prolonged health crises.

# 49 Significance Statement

49 Significance Statement50 Emotional bubbles during health crises represent critical threats to public mental health and 51 policy effectiveness, yet their formation mechanisms remain poorly understood. This study 52 reveals that emotional bubbles emerge from competition between stabilizing mainstream 53 media feedback and amplifying social media feedback, modulated by individual 54 psychological sensitivity thresholds. Using phase transition theory, we demonstrate that 55 emotional systems undergo sudden collapse when critical combinations of psychological 56 sensitivity and media composition are reached. These findings bridge physics, psychology, 57 and public health by providing quantitative predictions for emotional instability during crises. 58 Our framework enables policymakers to anticipate and prevent emotional bubble formation 59 before critical tipping points are reached, offering evidence- based strategies for maintaining 60 population mental health resilience during prolonged health emergencies.

# Introduction

The COVID- 19 pandemic created unprecedented health uncertainties, triggering widespread negative emotions and an infodemic of conflicting risk information that challenged public mental health worldwide(1- 3). Within this complex information ecology, a concerning phenomenon emerged: "emotional bubbles" - dynamic collective emotional states characterized by critical instability, where populations exhibit deceptive surface stability with high stickiness yet undergo rapid acceleration toward fragile collapse when critical thresholds are breached.[1] These formations significantly disrupt public health behaviors and policy effectiveness. In China, during the COVID- 19 pandemic, a highly representative phenomenon of social media "emotional bubbles" also emerged, therefore understanding the formation mechanisms of these is crucial for managing negative emotional responses during prolonged health and other types of crises.

The Social Amplification of Risk framework reveals how risk signals are amplified through information transmission and social response processes, particularly when risk events interact with psychological, social, and cultural processes that may significantly heighten or attenuate public risk perceptions and related behaviors, provides the theoretical foundation for our study on emotional bubble formation(4). Empirical studies on negative sentiment amplification during crises have identified multiple factors that contribute to emotional escalation, including political discontent and pandemic fatigue (5), overperception towards moral outrage (6, 7), real- time social media interactions (8- 10), and policy interventions affecting emotional expression patterns (1, 11). However, these studies predominantly analyze linear amplification processes driven by self- reinforcing feedback mechanisms (12, 13), treating emotional polarization as a monotonic escalation phenomenon. This approach dose not capture that emotional dynamics constitute inherently nonlinear processes governed by competing feedback mechanisms, where systems can exhibit complex behaviors including periods of apparent stability followed by sudden acceleration toward extreme states, rather than uniform linear progression.

Recently, a number of theoretical studies have pointed out that adaptive dynamics involving feedback mechanisms are responsible for opinion clustering and polarization processes in social systems. Multi- agent interaction analyses demonstrate that influencers promote short- term opinion clustering (14), while algorithmic recommendation systems create fragmented landscapes of transient clusters that remain vulnerable to disruption (15,

16). These dynamics are further complicated by the coupling effects between information source stance extremeness (17, 18), user openness, and algorithmic bias, which can produce diverse macroscopic behaviors ranging from manipulation to polarization and stabilization (19, 20). However, existing studies often treat the information environment as homogeneous, lacking insights into the co-evolution mechanisms between different information sources with fundamentally different feedback properties. Moreover, these studies neglect the psychological mechanisms underlying public responses, particularly overlooking how individual psychological sensitivity to risk information systematically drives selective exposure behaviors. Through these selective behaviors, the public actively seeks information that aligns with their psychological predispositions, creating self-reinforcing feedback loops that reshape the information environment itself (21). This gap is particularly significant given the distinct roles of mainstream media as “social anchors” versus We media actors who strategically adjust viewpoints to maximize followers (users with similar viewpoints) through polarization strategies, and that emotion information propagation follows step-flow patterns where government-verified accounts become more trustworthy under crisis circumstances (22, 23). The conflicting feedback mechanisms between stabilizing mainstream media and amplifying social media create complex co-evolutionary dynamics that produce nonlinear emotional bubble characteristics, which cannot be captured by single-factor feedback loop studies or homogeneous information environment simulations. Hence, the origin of emotional bubble dynamics in mixed-feedback information ecosystems with heterogeneous source motivations and bidirectional public-information feedback remains unknown. Therefore, this study develops that emotional bubble formation results from positive feedback amplification in social information channels.

Building on these studies, we hypothesize that emotional bubbles emerge from the interplay between internal psychological responses and external information environment dynamics, where differential feedback mechanisms create nonlinear critical states through their mutual interactions. This work investigates the formation mechanisms of emotional bubbles and implements mixed- feedback model (MFM), a quantitative framework, for understanding how feedback imbalances drive emotional concentration dynamics. Using sentiment analysis of 294,434 Weibo posts related to Long COVID, we identify differential roles of mainstream media and We media in emotional polarization: mainstream media risk exposure exhibits an inverted U- shaped relationship with emotional polarization, while We media demonstrates a direct linear correlation with intense emotional responses, revealing distinct feedback mechanisms that drive emotional concentration patterns. We develop mean- field theoretical equations incorporating two key feedback mechanisms: (1) internal psychological feedback through individual risk sensitivity thresholds (see Methods for  $\phi$  and  $\theta$  parameter definitions), where low risk exposure triggers apathetic responses while high exposure generates intense emotional reactions, and (2) external media feedback dynamics where mainstream media provides stabilizing negative feedback while We media creates amplifying positive feedback loops. Crucially, as individual psychological sensitivity

increases, people increasingly engage in selective exposure behaviors that systematically erode moderate emotional buffers, creating a mechanistic pathway from individual psychology to collective instability. Our mixed- feedback framework reveals three key mechanisms driving emotional bubble formation. First, heightened psychological sensitivity triggers selective information consumption that bypasses moderate content, fundamentally reshaping the information environment through bidirectional feedback loops. Second, competing media feedback mechanisms create qualitatively different system behaviors: mainstream media's stabilizing feedback maintains continuous transitions, while We media's amplifying feedback drives discontinuous jumps when critical thresholds are exceeded. Third, network simulations validate that progressive mainstream media removal triggers cascading transitions from resilient group stabilization to brittle system fragility, transforming robust emotional ecosystems into configurations prone to sudden collapse. This framework establishes emotional bubbles as threshold- mediated phenomena where the interaction between psychological sensitivity and media composition determines whether systems exhibit continuous resilient dynamics or discontinuous fragile responses.

# Results

# Empirical evidence of emotional polarization during the COVID-19 pandemic

To investigate the formation mechanisms of emotional bubbles during health crises, we analyzed three complementary datasets collected from Weibo, China's largest microblogging platform, during the COVID- 19 pandemic period (January 2020 - December 2023). The first dataset comprises 49,288 posts from verified We media accounts discussing COVID- 19 long- term effects, characterized by high emotional variability and opinion- driven content. The second dataset contains 36,480 posts from verified mainstream media outlets, representing authoritative information sources with more balanced emotional expressions. The third dataset encompasses 208,666 individual user posts expressing personal emotional responses to pandemic- related events, providing comprehensive coverage of public sentiment dynamics. These datasets collectively span the complete information ecosystem during the pandemic, capturing both elite discourse and grassroots emotional expressions across different phases of the health crisis.

To ensure reliable content analysis, we developed a multi- task machine learning framework for automated classification of emotional states and risk perceptions across our datasets. Through 100 independent training experiments with random train- test splits, our DistilBERT- based classifier achieved consistently high performance: emotion classification reached  $98.0\%$  test accuracy (CV:  $0.73\%$ ), mainstream media risk assessment achieved  $91.5\%$  accuracy (CV:  $1.82\%$ ), and We media risk classification attained  $96.7\%$  accuracy (CV:  $1.64\%$ ). These robust classification results across heterogeneous information sources validate the reliability of our subsequent empirical analysis (Figure 1 and Supplementary Figure 1).

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-27/9ca076a7-360a-4b46-9317-b4e5b349dce7/d6a86c82ca3cfc26f5e1f0e5344c824005173bfd113ae955cc3187ee8225035d.jpg)  
Figure 1 | Performance of classification. a, Probability density distributions of test accuracy across independent training runs for emotion classification (blue), mainstream media risk assessment (green), and We media risk evaluation (red). b, Box plots of F1-macro score distributions. Box plots show median (center line), interquartile range (box), and  $1.5 \times \mathrm{IQR}$  whiskers. c, Overview of mixed-feedback model (MFM). The three-layer framework illustrates dual feedback mechanisms in information ecosystems, where yellow layer represents mainstream media, green layer represents We media, and brown layer represents the public. Mainstream media provides stabilizing negative feedback (left loop) that counteracts extreme public states, while We media generates amplifying positive feedback (right loop) that responds to attention-seeking behaviors. These opposing feedback mechanisms determine system stability and emotional bubble formation.

In our framework, "risk information" refers to media reporting tendencies regarding the potential dangers of Long COVID health consequences. For example, when mainstream media interviews medical experts who conclude that Long COVID produces minimal long- term sequelae, this constitutes no- risk information; conversely, reports emphasizing severe persistent symptoms represent risk information. These risk signals are absorbed by the public and undergo psychological amplification or attenuation processes that drive emotional state transitions (see Supplementary Materials for detailed risk classification methodology).

To empirically investigate the formation mechanisms of emotional bubbles, we selected December 2022 time series data to examine the relationship between risk information and public emotional responses. This period represents the peak of Long COVID discussions in China and coincides with significant policy shifts by the Chinese government, providing an ideal natural experiment for studying emotional bubble dynamics (24). We aggregated the

classified data into daily- level representations, where each data point represents the distributional state of emotions and risk perceptions within a single day. This temporal aggregation enables us to capture the dynamic interplay between information exposure and collective emotional responses at the population level.

Figure 2a displays the temporal evolution of risk information from both mainstream media and We media sources, alongside the corresponding emotional arousal patterns across three distinct intensity levels: high arousal (representing anger and sarcasm) (25, 26), middle arousal (reflecting stress, alertness, and anxiety) (27), and low arousal (indicating indifference, apathy, coldness, and depression) (see Supplementary Section 2 for detailed keyword classifications) (28- 30). The time series reveals notable fluctuations in risk information dissemination, with We media exhibiting higher volatility compared to mainstream media, particularly during critical periods of the pandemic progression. Most importantly, we observed a striking consistency between high arousal emotional states and We media risk information quantity, with a significant positive Pearson correlation  $(r = 0.387$ $p = 0.042)$ . In stark contrast, mainstream media showed no significant linear correlation with emotional states  $(r = 0.036$ $p = 0.854)$ , providing empirical evidence for media differentiation: We media exhibits direct positive feedback amplification, while mainstream media demonstrates more complex, non- linear feedback dynamics.

Our analysis reveals distinct relationships between information exposure and emotional dynamics that precisely validate the theoretical predictions of our mixed- feedback model (MFM). Figure 2b demonstrates the relationship between risk information exposure and emotional concentration, measured as 1- entropy to quantify the degree of emotional clustering. We quantify emotional concentration using the entropy measure

$S = - \sum_{i = 1}^{3}\sqcup p_{i}\log_{2}p_{i},$  where  $p_i$  represents the relative frequency of emotional state  $i$  (low,

middle, high arousal) within each daily observation. The emotional concentration index is then calculated as  $C = 1 - S / S_{max}$ , where  $S_{max} = \log_2 3$  represents the maximum possible entropy for three emotional states. The relationship exhibits a pronounced U- shaped pattern, indicating that moderate levels of risk exposure promote more balanced emotional distributions across the population. Conversely, both minimal risk exposure and excessive risk information influx drive emotional concentration, creating conditions conducive to emotional bubble formation. Consistent with previous research on digital echo chambers and emotional contagion dynamics (31, 32), this empirical finding provides quantitative validation that balanced information environments maintain emotional diversity, while extreme information conditions systematically facilitate clustering dynamics through threshold- mediated mechanisms.

Disaggregated media- type analyses reveal the mechanistic foundations underlying our MFM framework through distinct influence patterns that validate theoretical predictions. Mainstream media risk exposure demonstrates a characteristic inverted U- shaped relationship with emotional polarization, quantified as  $P = \left(p_{high} + p_{low}\right) / p_{middle}$  where  $p_i$  represents the

proportion of emotional state  $i$  within daily observations (Figure 2c). This curvilinear pattern reveals that moderate messaging inconsistency maximizes emotional polarization, while both minimal reporting and highly coherent risk communication reduce extreme emotional responses. The inverted U- curve provides compelling empirical evidence for mainstream media's stabilizing negative feedback role: when mainstream media maintains appropriate levels of reporting consistency, it acts as a regulatory force that effectively reduces emotional clustering in the population (33), precisely as predicted by our MFM framework.

We media platforms exhibit fundamentally contrasting dynamics that validate the positive feedback amplification component. Figure 2d reveals a robust monotonic relationship between We media risk exposure and emotional polarization, demonstrating systematic amplification of emotional clustering through self- reinforcing feedback loops. This linear escalation pattern spans the entire exposure range, indicating that We media platforms lack the regulatory mechanisms inherent in mainstream media and instead function as pure amplification systems that intensify emotional signals proportionally to their magnitude. The sustained linear increase provides direct empirical validation of the positive feedback amplification mechanisms, establishing that We media creates self- reinforcing cycles that systematically drive emotional bubble formation through fundamentally different mechanistic pathways than mainstream media channels.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-27/9ca076a7-360a-4b46-9317-b4e5b349dce7/abae450a233a6b15a4c7f3ec09cdbc5091784e9928065715defc23f6f55e2205.jpg)  
Figure 2 | Empirical evidence of emotional bubble dynamics during COVID-19. a,

Temporal evolution of risk information from mainstream media (purple dotted) and We media (red dotted), with emotional arousal patterns for high (orange dashed). b, U- shaped relationship between risk exposure and emotional concentration. Shaded areas represent  $95\%$

confidence intervals of the fitted regression curve. c, Inverted U- shaped relationship between mainstream media exposure and emotional polarization. d, Linear positive relationship between We media exposure and emotional polarization.

These empirical findings provide compelling validation of our theoretical predictions from the introduction, confirming that emotional dynamics constitute inherently nonlinear processes governed by competing feedback mechanisms rather than uniform linear progression. Our analysis reveals a fundamental nonlinear relationship between risk information exposure and public emotional dynamics, as evidenced by the pronounced U- shaped pattern between total risk exposure and emotional concentration. Importantly, we discovered differential impacts of heterogeneous information sources on public extreme emotions, directly validating the media competition mechanisms theorized in our introduction. We media exhibits a significant linear influence on high arousal states, providing direct empirical evidence for positive feedback amplification mechanisms, while mainstream media demonstrates complex non- linear correlations indicative of regulatory feedback architecture. These distinct feedback patterns directly support our theoretical equations (Eq. 4 and Eq. 5) in the Methods section, where media competition dynamics are formalized as opposing positive and negative feedback mechanisms within our mixed- feedback framework.

# Theoretical framework reveals mixed-feedback mechanisms driving

# emotional bubble formation

Our theoretical framework predicts emotional bubble formation through systematic analysis of psychological threshold interactions with media feedback dynamics. We developed self- consistent equations incorporating dual feedback mechanisms: internal psychological thresholds ( $\phi$  and  $\theta$  parameters) govern individual emotional responses, while external media dynamics create opposing feedback loops—mainstream media provides stabilizing negative feedback, whereas We media generates amplifying positive feedback. Here,  $\phi$  represents the low arousal threshold—the minimum risk information exposure required to trigger apathetic responses, while  $\theta$  represents the high arousal threshold—the critical exposure level that induces intense emotional reactions (see Methods for detailed mathematical formulations). These psychological sensitivity parameters capture individual differences in risk information processing: smaller  $\theta - \phi$  differences indicate higher psychological sensitivity, where individuals more readily transition between extreme emotional states.

The system responsiveness analysis reveals a fundamental relationship between psychological thresholds and system vulnerability to emotional clustering. We quantify emotional responsiveness as  $R = \dot{c}\frac{dX_H}{dr}\vee \dot{c}$ , measuring the absolute rate of change in high- arousal population proportions ( $X_H$ ) relative to mainstream media removal ratios ( $r$ ), where  $r$

represents the fraction of stabilizing mainstream media sources systematically removed from the information ecosystem (see Methods for complete variable definitions). Figure 3a demonstrates that higher  $\phi$  values systematically amplify emotional responsiveness to information changes. The circular packing visualization quantifies this relationship: larger circles correspond to higher  $\phi$  values, indicating that more sensitive populations become dramatically more vulnerable to emotional bubble formation when exposed to identical information environments. This vulnerability emerges through a specific behavioral mechanism: as psychological thresholds  $\phi$  and  $\theta$  converge (smaller  $\theta - \phi$  differences), individuals increasingly engage in selective exposure behaviors, bypassing moderate or contradictory "buffering" information to directly consume extreme or homogeneous content. This psychological tendency drives selective contact behaviors that systematically erode the moderate emotional buffer  $(X_{M})$ , establishing psychological sensitivity as the primary driver of system fragility (Figure 3b).

As  $\phi$  increase, the emotional landscape undergoes systematic reorganization characterized by the disappearance of moderate states. Medium arousal proportions  $(X_{M})$  compress while extreme states dominate (Figure 3b- c), creating binary configurations where populations oscillate between high and low arousal without stabilizing intermediate states. We quantified this instability using jump amplitudes  $J = \max \left(X_{H}^{\text{window}}\right) - \min \left(X_{H}^{\text{window}}\right)$  at critical transition points (Figure 3d). Higher  $\phi$  values produce systematically larger discontinuous jumps, revealing why emotional bubbles exhibit surface stability yet sudden collapse—they represent critical systems operating near phase transition boundaries where compressed emotional systems lack buffering capacity to absorb perturbations smoothly.

The progressive removal of mainstream media triggers cascading instability that amplifies these underlying vulnerabilities. As mainstream media removal ratios increase beyond critical thresholds, negative feedback mechanisms weaken systematically, causing rapid differentiation toward extreme emotional states. This effect intensifies with higher  $\phi$  values, demonstrating that psychological sensitivity and media composition interactions fundamentally determine whether systems exhibit continuous resilient dynamics or discontinuous fragile responses. When both psychological sensitivity increases and stabilizing media influence decreases, systems transition from robust configurations to inherently unstable emotional ecosystems characterized by the elimination of moderate emotional buffers.

This transition from stability to fragility exhibits universal characteristics across the psychological threshold spectrum, as demonstrated through detailed analysis of emotional state trajectories. Figure 3b presents these dynamics for various  $\phi$  values under fixed  $\theta = 0.49$ , revealing systematic stickiness patterns that expose the deceptive nature of apparent stability in emotional systems. At low mainstream media removal ratios  $(r< 0.6)$ , all emotional states exhibit minimal variation regardless of psychological sensitivity parameters, displaying high stickiness that creates the illusion of robust system stability. However, as mainstream media presence continues to diminish, all parameter configurations inevitably reach critical

transition points where this stickiness collapses and emotional states become increasingly fragile and responsive to perturbations.

The psychological sensitivity parameter  $\phi$  fundamentally modulates this transition from stable to fragile dynamics. For  $\phi \leq 0.41$ , systems maintain relatively gradual transitions with later critical points, preserving some resilience against media composition changes. Conversely, when  $\phi > 0.41$ , critical points arrive earlier and transitions become more abrupt, reflecting accelerated fragility that makes systems vulnerable to rapid emotional reorganization (See Figure 3b). This threshold- dependent acceleration directly connects to the microscopic transition dynamics revealed in our network simulations, higher  $\phi$  values systematically increase direct transitions between extreme emotional states ( $X_{H}$  and  $X_{L}$ ), bypassing the stabilizing medium arousal buffer and creating the mechanistic foundation for both the early onset and increased amplitude of fragile responses observed in our theoretical predictions (see Figure 6e). These convergent findings establish that emotional bubble formation represents a threshold- mediated phenomenon where seemingly stable emotional environments undergo rapid reorganization when critical combinations of psychological sensitivity and media composition are reached. The systematic erosion of negative feedback mechanisms drives this universal transition from resilient to fragile emotional ecosystems, with psychological sensitivity determining the specific dynamics and timing of system collapse during crisis periods.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-27/9ca076a7-360a-4b46-9317-b4e5b349dce7/cf17a464b9e870b82f3839780159b7e1673059531283163eab6a2301789ddd38.jpg)  
Figure 3 | Individual psychological thresholds drive emotional bubble formation through mixed-feedback mechanisms. a, Emotional responsiveness distribution across varying

individual psychological thresholds  $(\phi)$  b,c, Three- state emotional distribution evolution across removal ratios, demonstrating systematic compression of medium arousal states. d, Jump amplitude intensification with increasing risk sensitivity across parameter space.

# Phase transition analysis reveals critical tipping points in emotional

# bubble formation

How do empirically observed emotional bubbles emerge from psychological- media interactions? We address this through systematic phase transition analysis of our mean- field theoretical framework, identifying critical tipping points where emotional ecosystems transition from resilient to brittle configurations. Our theoretical analysis reveals the precise mechanisms driving systems toward fragile, bubble- like states through comprehensive mapping of the psychological threshold parameter space.

The phase diagram in psychological threshold space  $(\phi , \theta)$  maps three distinct behavioral regimes (Figure 4a). The large dark teal region represents stability, where systems respond smoothly without abrupt shifts. Two critical regions emerge: a blue region near the  $\phi = \theta$  diagonal indicates first- order phase transitions—discontinuous jumps representing emotional bubble collapse. A thin purple region shows second- order phase transitions with continuous but critical changes. Crucially, this second- order transition region serves as an early warning boundary that separates stable system dynamics from catastrophic fragility zones. At these critical points, systems maintain continuous responses but exhibit maximum sensitivity to perturbations, manifesting as diverging correlation lengths and power- law scaling behaviors that signal impending instability. The concentration of transitions near  $\phi = \theta$  reveals a fundamental principle: as the psychological gap between apathy and anger narrows, systems become inherently unstable and prone to dramatic shifts, with the second- order boundary marking the precise threshold where resilient emotional ecosystems begin their transition toward brittle configurations vulnerable to sudden collapse.

This instability mechanism becomes particularly evident at fixed  $\theta = 0.49$  (Figure 4b). As the low- arousal threshold  $\phi$  increases, the system's response to mainstream media removal fundamentally changes. For  $\phi < 0.41$ , high- arousal proportions  $(X_H)$  decrease continuously, indicating resilient dynamics. At the critical point  $\phi = 0.41$ , the system undergoes a second- order phase transition—a pivotal boundary condition where the system maintains continuous responses while developing critical sensitivity that forewarns of approaching instability. This critical point serves as a predictive threshold: systems operating at this boundary exhibit maximum responsiveness to environmental changes while retaining stability, providing an early detection mechanism for emotional bubble formation. For  $\phi > 0.41$ , the dynamics shift entirely:  $X_H$  first climbs then abruptly collapses, signifying discontinuous first- order transitions. This demonstrates that increasing public sensitivity renders stable emotional ecosystems fragile and susceptible to sudden collapse, with the second- order transition point

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-27/9ca076a7-360a-4b46-9317-b4e5b349dce7/a14e7baddd7ae2d45e090025ddb920025e25906dda8bf146c09339b0aeefbaed.jpg)  
Figure 4 | Phase diagram reveals critical transitions in emotional dynamics across

Figure 4 | Phase diagram reveals critical transitions in emotional dynamics across psychological threshold space. a, The main panel illustrates the comprehensive phase diagram in the  $(\phi , \theta)$  parameter space. The x- axis represents the low arousal threshold  $(\phi)$ , while the y- axis represents the high arousal threshold  $(\theta)$ , both ranging from 0.15 to 0.55. Three distinct behavioral regimes are identified: the dark teal region indicates the absence of a phase transition, the blue region denotes a first- order phase transition characterized by discontinuous shifts in emotional states, and the thin striped purple region represents a second- order phase transition where critical changes occur. The diagonal line marks the fundamental boundary where  $\phi = \theta$ . b, The inset provides a magnified view of the critical transition region at a fixed  $\theta = 0.49$ . It showcases the evolution of the state variable  $X_H$  as a function of the mainstream media removal ratio  $(r)$  for various  $\phi$  values. The inset reveals that as  $\phi$  increases, the system transitions from a robust continuous decrease in  $X_H$  (for  $\phi < 0.41$ ) to a critical second- order transition  $(\phi = 0.41)$  and, ultimately, to a discontinuous first- order jump (for  $\phi \geq 0.43$ ), demonstrating the fundamental role of psychological thresholds in shaping emotional bubble formation.

Critical phenomena analysis confirms these distinct transition types (Figure 5). At the second- order critical point  $(\phi = 0.41)$ , the correlation length  $\xi$  diverges at the critical removal ratio  $r_c$ , indicating long- range correlations throughout the system (Figure 5a). Power- law scaling emerges with critical exponents  $\nu_i = 0.935$  and  $\nu_{i = 0.940i}$  (Figure 5b), providing quantitative evidence of second- order transitions. In contrast, first- order transitions at higher  $\phi$  values exhibit sudden shifts in emotional concentration  $C = 1 - S / S_{max}$ , where entropy  $S = - \sum_{i = 1}^{3} \square p_i \log_2 p_i$  (Figure 5c). These transitions reflect abrupt reorganization of the entire emotional landscape.

The mechanistic foundation of this stability- to- fragility transition lies in moderate emotional state compression  $(X_M)$ . Parameters  $\phi$  and  $\theta$  represent public sensitivity to risk information; smaller differences  $\theta - \phi$  indicate higher sensitivity, reducing moderate responses and driving individuals toward extreme arousal states (Figure 5d). This erosion of emotional buffers explains system brittleness: without moderately aroused populations to absorb fluctuations, emotional ecosystems become polarized and fragile, creating bubbles that appear stable but collapse suddenly at tipping points.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-27/9ca076a7-360a-4b46-9317-b4e5b349dce7/fc49b22393694fee03a768063fadc88da44b0725eb250b0a9a665104b8bd7acc.jpg)

# Figure 5 | Critical phenomena and state-space compression at phase transitions. a,

Correlation length  $\xi$  as a function of removal ratio  $r$  . The divergence of  $\xi$  at the critical point for  $\phi = 0.41$  (orange curve) is a hallmark of a second- order transition. b, Power- law scaling of the correlation length at the second- order critical point  $(\phi = 0.41,\theta = 0.49)$  . The linear relationship on a log- log plot of  $\xi$  versus the distance from the critical point,  $\dot{\iota} r - r_{c}\vee \dot{\iota}$  confirms a power- law behavior  $\xi \sim \vee r - r_{c}\dot{\iota}^{\mathrm{- }\nu}$  . Linear fits (dashed lines) yield the critical

exponents  $\nu = 0.940$  for the left side  $(r< r_{c})$  and  $\nu = 0.935$  for the right side  $(r > r_{c})$  . c, Emotional concentration (1- entropy) across different parameter regimes. Curves corresponding to first- order transitions exhibit a discontinuous jump, in stark contrast to the continuous changes observed in other regimes. d, Compression of the  $X_{M}$  proportion. The line plot (left) shows that a sharp dip near the transition point. This contrasts with the smoother declines in other parameter regimes. The heatmap (right) further visualizes this across the  $(\phi ,r)$  space, where greener regions represent a larger proportion of  $X_{M}$  and show its compression as a function of both psychological threshold and media removal.

# Network simulations validate microscopic dynamics and transition

# mechanisms

To validate our mean- field theoretical predictions and capture individual- level dynamics that drive emotional bubble formation, we implemented agent- based network simulations where each node represents an individual with distinct emotional states and threshold- based decision rules. Our simulation framework models three types of agents: mainstream media sources, We media sources, and public individuals, connected through a directed network that captures realistic information flow patterns. Each public agent transitions between emotional states based on local risk exposure from connected media sources, following the same threshold dynamics (  $\phi$  and  $\theta$  parameters) as our theoretical model. This microscopic approach enables us to study emergent collective behaviors arising from individual psychological responses and examine transition mechanisms that cannot be captured by mean- field approximations alone.

Network simulations provide microscopic validation of emotional bubble formation mechanisms, achieving excellent agreement with mean- field theoretical predictions (Figure 6b- c). Low Root Mean Square Error (RMSE) between simulated and theoretical steady- state solutions confirms simulation fidelity. The simulations reveal systematic polarization patterns (defined as  $1 - X_{M}$ ); systems become strongly polarized when psychological thresholds  $\phi$  and  $\theta$  converge or when mainstream media removal ratio  $r$  increases (Figure 6a).

The psychological sensitivity parameter  $\phi$  fundamentally modulates the transition from stable to fragile dynamics at the microscopic level. For  $\phi \leq 0.41$ , individual agents maintain relatively gradual state transitions with later critical points, preserving system resilience against media composition changes. Conversely, when  $\phi > 0.41$ , critical points arrive earlier and individual transitions become more abrupt, reflecting accelerated fragility that makes the entire system vulnerable to rapid emotional reorganization (Figure 3b). Analysis of direct

transition dynamics (Figure 6e) reveals that higher  $\phi$  values systematically increase transitions between extreme emotional states (  $X_{H}$  and  $X_{L}$  ), bypassing the stabilizing medium arousal buffer. This erosion of emotional buffers creates the mechanistic foundation for emotional bubble formation, where seemingly stable emotional environments undergo rapid reorganization when critical combinations of individual psychological sensitivity and media composition are reached.

The critical role of negative feedback mechanisms emerges through polarization metric analysis (Figure 6d). High polarization intensifies with increasing  $r$ , approaching unity as  $r \to 1$ , providing compelling evidence that progressive negative feedback erosion drives system instability and emotional bubble formation.

To understand why this feedback proves so critical, we compute direct transition rates—the proportion of individuals switching directly between high- arousal  $(X_{H})$  and low- arousal  $(X_{L})$  states, bypassing moderate  $(X_{M})$  buffers. This metric quantifies system stability against extreme oscillations. Our analysis reveals that transition rates systematically increase with psychological threshold  $\phi$ , explaining why heightened public sensitivity exacerbates polarization (Figure 6e). The relationship between transition rates and removal ratio  $r$  traces a distinct U- shaped curve (Figure 6f), revealing fundamental system dynamics shifts. Initially, at  $r = 0$ , competing positive and negative feedback creates volatile environments with frequent state transitions. As negative feedback progressively diminishes  $(0 \leq r \leq 0.762)$ , unopposed positive feedback enables stable polarized group formation, reducing overall transition rates as individuals become locked in states. However, once stabilizing negative feedback becomes nearly absent  $(r \geq 0.8)$ , systems become highly unstable. Without opposing forces to dampen perturbations, random fluctuations amplify through unopposed positive feedback loops, triggering large- scale population shifts between extremes. This U- shaped curve demonstrates that removing negative feedback first stabilizes polarized groups, then destabilizes entire systems.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-27/9ca076a7-360a-4b46-9317-b4e5b349dce7/0ba4173eadb8d1e5a0416d912f4db1fe34ec5d3f7d4c63f20545afae344cae52.jpg)

Figure 6 | Validation and microscopic dynamics of network simulations. a, A 3D bubble plot from network simulations illustrating the polarization level (color bar) across the  $(\phi , \psi , r)$  parameter space. b,c, Validation of the agent- based simulation against the self- consistent mean- field theory. b, The Root Mean Square Error (RMSE) between the simulated and theoretical steady- state proportions of  $X_{H}$  and  $X_{L}$  is consistently low across the  $\phi$  parameter range. c, The probability density distribution of all RMSE values, showing a narrow distribution centered on a low error value, which confirms the high fidelity of the simulation. d, Boxplots of two polarization metrics, High Polarization and Total Polarization, as a function of the mainstream media removal ratio  $r$ . Box plots show median (center line), interquartile range (box), and  $1.5 \times \mathrm{IQR}$  whiskers. e,f, Direct transition rates between extreme emotional states ( $X_{H}$  and  $X_{L}$ ). e, The rate of direct transitions increases monotonically with the psychological threshold parameter  $\phi$ . f, The direct transition rate exhibits a U- shaped relationship with the removal ratio  $r$ .

# 506 Discussion

Understanding how emotional bubbles emerge during health crises represents a critical challenge for managing public mental health and policy effectiveness, with increasing risks of prolonged pandemics, climate- induced disasters, and rapid technological shifts in information ecosystems. Such health and social shocks could induce substantial shifts in collective emotional states by altering the psychological and informational mechanisms that govern emotional responses in populations. A substantial body of research has focused on modeling the amplification effects of negative sentiment during crises (34- 37); however, there has been limited investigation into the dynamic formation mechanisms of emotional bubbles—self- reinforcing emotional states that exhibit surface stability yet inherent fragility leading to sudden collapse. The underlying psychological and informational factors that drive such emotional clustering dynamics during health crises are not well understood. Motivated by these critical challenges, we used empirical sentiment data from Weibo posts and developed a mixed- feedback theoretical framework to quantitatively measure and analyze the formation mechanisms of emotional bubbles during the COVID- 19 pandemic.

In this context, our study contains three key findings toward understanding emotional bubble dynamics in mixed- feedback information ecosystems. First, our empirical analysis reveals that the relationships between risk information exposure and emotional polarization are highly complex products of heterogeneous media feedback mechanisms rather than simple linear amplification processes. We observe differential roles of mainstream media and We. media: mainstream media risk exposure exhibits an inverted U- shaped relationship with emotional polarization, providing stabilizing negative feedback that moderates extreme responses, while We media demonstrates a direct linear correlation with high arousal states, creating amplifying positive feedback loops. Most importantly, we discover that emotional concentration follows a U- shaped pattern with respect to total risk information exposure, indicating that moderate levels of risk information promote balanced emotional distributions, while both minimal and excessive exposure drive emotional clustering. These findings

demonstrate that emotional bubble formation emerges from imbalanced feedback environments rather than uniform information saturation. Second, using our self- consistent mean- field theoretical framework, we showed that incorporating mixed- feedback mechanisms with individual psychological thresholds ( $\phi$  and  $\theta$  parameters) substantially improves the predictability of emotional bubble formation compared to traditional linear polarization models. Our theoretical predictions successfully capture the empirically observed nonlinear relationships, revealing that emotional bubbles arise from the compression of moderate emotional states as psychological sensitivity increases. Third, phase transition analysis reveals that the system undergoes qualitatively different dynamics across the psychological threshold space  $(\phi , \theta)$ , transitioning from continuous, resilient responses to discontinuous, fragile configurations characterized by sudden collapse when critical thresholds are breached.

Our study has several limitations that suggest important directions for future research. First, for model simplicity and theoretical tractability, we did not incorporate differential weighting mechanisms between mainstream media and We media sources, treating their influence as proportional to their quantity rather than their actual reach or credibility. We could further differentiate media influence by incorporating audience size, engagement metrics, and trust levels to better capture the heterogeneous impact of different information sources. Moreover, following recent studies focusing on the substantial differences in emotional responses across sociodemographic groups(38, 39), we could decompose the psychological thresholds into different demographic segments to better understand which populations are most vulnerable to emotional bubble formation. Second, for theoretical parsimony, we focused exclusively on risk information feedback while neglecting other environmental factors that influence emotional dynamics, such as social network homophily (40- 42), peer influence (43), and offline community interactions (44). The emotional bubble formation process likely involves complex interactions between information exposure and social contagion mechanisms that our current framework does not capture. Incorporating network- based social influence models with our mixed- feedback framework could provide a more comprehensive understanding of emotional clustering dynamics.

Our findings have significant implications for both theoretical understanding and practical management of collective emotional responses during health crises. While previous research on emotional polarization has predominantly focused on linear amplification processes and static cross- sectional analyses, our results demonstrate that emotional bubble formation—characterized by nonlinear transitions and inherent instability—represents the key mechanism for understanding collective emotional dynamics during prolonged crises. Traditional polarization research assumes gradual, monotonic shifts driven by group confrontation, but our mixed- feedback framework reveals that emotional systems can undergo rapid, discontinuous transitions from apparently stable states to sudden collapse when critical psychological or informational thresholds are exceeded. Through our theoretical predictions and network simulation validations, we establish that these nonlinear dynamics originate from the competition between stabilizing negative feedback (mainstream media) and amplifying positive feedback (We media) mechanisms, creating conditions where small perturbations can trigger system- wide emotional reorganization. For policy implications, our framework

highlights the critical importance of maintaining balanced information ecosystems rather than implementing uniform intervention strategies. The existence of critical transition points suggests that policy interventions must be carefully calibrated to account for the nonlinear sensitivity of emotional systems—removing too much mainstream media stabilization or allowing excessive We media amplification can push the system beyond critical thresholds, triggering sudden emotional bubble formation that becomes extremely difficult to reverse. This understanding provides a theoretical foundation for developing adaptive intervention strategies that can anticipate and prevent emotional bubble formation before critical points are reached, ultimately enhancing the resilience of public mental health systems during prolonged health crises.

# 585 Materials and Methods

To investigate the formation mechanisms of emotional bubbles in mixed- feedback information ecosystems, we employed a three- pronged methodological approach combining automated content analysis, mean- field theoretical modeling, and computational simulations. This integrated framework enables us to systematically examine the interplay between psychological feedback mechanisms and media information dynamics that drive emotional concentration phenomena.

# 592 Automated Content Analysis

We collected three complementary datasets from Weibo, China's largest microblogging platform, spanning the COVID- 19 pandemic period (January 2020 - December 2023). The datasets comprise 49,288 posts from verified We media accounts, 36,480 posts from verified mainstream media outlets, and 208,666 individual user posts expressing personal emotional responses to pandemic- related events. These datasets collectively capture the complete information ecosystem during the health crisis, representing both elite discourse and grassroots emotional expressions across different phases of the pandemic.

Text preprocessing involved systematic noise removal, including elimination of special characters, emojis, and extremely short texts (less than 10 characters). We implemented standardized cleaning procedures to ensure data quality while preserving semantic content essential for emotional and risk assessment classification.

For supervised learning model development, we randomly sampled 5,000 posts from the combined dataset and distributed them among three trained volunteers for manual annotation. Each post was independently labeled for both emotional arousal levels (high, middle, low) and risk perception (risk, no- risk) following established coding protocols.

The classification framework builds upon Russell's circumplex model of affect (45), particularly its application to health crisis communication. Emotional content was categorized based on arousal levels:

- High-arousal emotions: Include anger, fear, criticism, and sarcasm with terms such as "anti-science," "hypocritical," "unscrupulous," "intimidation," and "bias"- Medium-arousal emotions: Exist at the threshold between high and low arousal, with phrases like "don't underestimate," "don't panic," and "stop intimidating"- Low-arousal emotions: Cover uncertainty, avoidance, confusion, sadness, helplessness, and indifference

Risk perception classification distinguished between posts emphasizing potential dangers of Long COVID health consequences (risk) versus those minimizing or dismissing such concerns (no- risk). Risk posts typically featured content emphasizing severe persistent symptoms, long- term health complications, or medical uncertainties, while no- risk posts highlighted recovery stories, minimal sequelae, or expert reassurances about limited long- term effects.

After quality control procedures and removal of inconsistent annotations, we retained 4,086 high- quality labeled samples for model training and validation.

We employed DistillBERT, a lightweight variant of BERT, as our base model for fine- tuning on the annotated dataset(46). The model architecture was optimized for multi- task learning, simultaneously predicting emotional arousal levels and risk perception categories.

To ensure robust performance evaluation, we conducted 100 independent random experiments, each with different train- test splits, generating comprehensive performance distributions across multiple evaluation metrics. The experimental results demonstrate excellent model performance across all classification tasks (Figure 1 and Supplementary Figure 1). Test accuracy achieved remarkable stability: emotion classification maintained  $95.8\%$  accuracy, mainstream media risk assessment achieved  $91.5\%$  accuracy, and We media risk classification reached  $88.8\%$  accuracy. These high- performance metrics validate our automated content analysis framework's capability to systematically quantify emotional polarization dynamics across heterogeneous information sources.

# Mean-Field Theoretical Modeling

We developed self- consistent equations to model the dynamic interactions between two core mechanisms: (1) public risk sensitivity representing individual psychological responses to risk information, and (2) positive- negative feedback imbalances in external information environments, where negative risk information gets amplified while positive information is suppressed. Our theoretical framework employs a three- state threshold dynamics model where individuals transition between high arousal, medium arousal, and low arousal emotional states, with the corresponding population fractions denoted as  $X_{H}$ ,  $X_{M}$ , and  $X_{L}$  respectively, subject to the conservation constraint:

$$
X_{H} + X_{M} + X_{L} = 1
$$

The emotional state variables  $X_{H}, X_{M}$ , and  $X_{L}$  represent the fractional populations in each arousal category, where  $X_{H}$  quantifies the proportion experiencing intense emotional responses such as anger and fear,  $X_{M}$  captures individuals in transitional states reflecting stress and anxiety, and  $X_{L}$  describes those exhibiting apathetic or withdrawn behaviors including indifference and depression. The threshold parameters  $\phi$  (low arousal threshold) and  $\theta$  (high arousal threshold) govern state transitions, where  $0 \leq \phi < \theta \leq 1$ . These psychological sensitivity parameters capture individual differences in risk information processing, with  $\phi$  representing the minimum risk exposure required to trigger apathetic responses and  $\theta$  denoting the critical exposure level that induces intense emotional reactions. Smaller differences  $\theta - \phi$  indicate higher psychological sensitivity, where individuals more readily transition between extreme emotional states without stabilizing in moderate arousal conditions.

The emotional state distributions are computed through threshold- based probabilistic transitions that reflect individual exposure to heterogeneous information sources. For individuals with input degree  $k$ , representing their connectivity to media sources, the probability of receiving  $s$  risk signals follows a binomial distribution that captures the stochastic nature of information exposure. The high arousal fraction is calculated as:

$$
X_{M} = \sum_{k}\sqcup P(k)\sum_{s = I\theta kJ}^{k}\sqcup \binom{k}{s}p_{r i s k}^{s}(1 - p_{r i s k})^{k - s} \tag{2}
$$

Similarly, the low arousal fraction is:

$$
X_{L} = \sum_{k}\sqcup P(k)\sum_{s = 0}^{l\phi kJ}\sqcup \binom{k}{s}p_{r i s k}^{s}(1 - p_{r i s k})^{k - s} \tag{3}
$$

where  $P(k)$  represents the degree distribution following a Poisson distribution with mean  $\langle k \rangle = k_{out}^{mainstream} + k_{out}^{wemedia}$ , reflecting the average connectivity of public individuals to both mainstream media and We media sources, and the medium arousal fraction is determined by  $X_{M} = 1 - X_{H} - X_{L}$ .

The system incorporates distinct feedback mechanisms for different media types that capture their contrasting roles in emotional amplification versus stabilization. Mainstream media risk perception follows a stabilizing feedback formula:

$$
p_{r i s k}^{m a i n s t r e a m} = \frac{1 - X_{H} + X_{L}}{2}
$$

672

This negative feedback mechanism reflects mainstream media's tendency to provide balanced reporting that counteracts extreme public emotional states, where higher public anxiety  $(X_{H})$  triggers more reassuring coverage while widespread apathy  $(X_{L})$  prompts

increased risk awareness messaging. In contrast, We media risk perception exhibits amplifying behavior:

$$
P_{r i s k}^{w e m e d i a} = X_{H}
$$

This positive feedback mechanism captures We media's attention- seeking dynamics, where platforms amplify high- arousal content to maximize engagement, creating selfreinforcing cycles that intensify emotional clustering. The overall risk perception integrates both media types weighted by their effective presence after removal:

683

$$
p_{r i s k} = \frac{p_{r i s k}^{m a i n s t r e a m}\cdot n_{m}\cdot(1 - r_{m}) + p_{r i s k}^{w e m e d i a}\cdot n_{w}\cdot(1 - r_{w})}{n_{m}\cdot(1 - r_{m}) + n_{w}\cdot(1 - r_{w})}
$$

where  $n_m$  and  $n_w$  represent the numbers of mainstream and We media sources in the information ecosystem, while  $r_m$  and  $r_w$  denote the respective removal ratios representing the fraction of each media type systematically removed from the system. These removal ratios enable systematic exploration of how media composition changes affect emotional bubble formation, with  $r = r_m$  serving as the primary control parameter in our analysis when We media presence remains constant  $(r_w = 0)$ .

The system reaches equilibrium through iterative solution of the coupled equations, where media risk calculations are integrated into overall risk perception, which then drives public emotional state updates followed by emotional state removal and renormalization to maintain conservation. This iterative process continues until convergence is achieved, typically within 100- 1000 iterations with tolerance  $\epsilon < 10^{- 6}$ , ensuring that the system reaches a stable equilibrium state where all coupled variables satisfy the self- consistent equations simultaneously.

To identify phase transitions and critical behavior, we employed Jacobian matrix analysis for stability assessment. The Jacobian matrix  $J$  is computed through numerical differentiation of the self- consistent equations:

$$
J_{ij} = \frac{\partial F_i}{\partial X_j}
$$

where  $F_i$  represents the  $i$ - th component of the self- consistent equation system. The eigenvalues  $\lambda$  of  $J$  determine the system's stability: negative real parts indicate stability, while positive real parts signal instability and potential bubble formation. We computed the correlation length  $\xi$  as the inverse of the maximum real eigenvalue:

$$
\xi = \frac{1}{|\lambda_{max}|}
$$

Near critical points,  $\xi$  exhibits power- law scaling behavior:

$$
\xi \sim \vee r - r_{c}\dot{c}^{-\nu} \tag{9}
$$

where  $r_c$  is the critical removal ratio and  $\nu$  is the critical exponent characterizing the universality class of the phase transition. This scaling analysis provides quantitative criteria for predicting emotional bubble formation thresholds and understanding the system's critical behavior.

# 711 Computational Network Simulations

To validate our theoretical framework and capture microscopic transition dynamics, we implemented agent- based network simulations where each node represents an individual with distinct emotional states and risk perception mechanisms. The network consists of three node types: mainstream media sources ( $n_m$  nodes), We media sources ( $n_w$  nodes), and public individuals ( $n_p$  nodes), with directed edges representing information flow from media sources to public individuals. Each public node  $i$  has an in- degree  $k_i$  following a Poisson distribution with mean  $\langle k \rangle = k_{\text{out}}^{\text{mainstream}} + k_{\text{out}}^{\text{wemedia}}$ , representing the number of media sources it receives information from, while its emotional state is characterized by a three- state variable  $X_i \in \{H, M, L\}$  (high, medium, low arousal).

The core innovation of our simulation approach lies in the individual- level threshold dynamics, where each public node transitions between emotional states based on its local risk exposure. For a node with in- degree  $k_i$ , the probability of receiving a risk signal follows a binomial distribution  $B(k_i, p_{\text{risk}}^{\text{local}})$ , where  $p_{\text{risk}}^{\text{local}}$  is the weighted average risk perception of its connected media sources. State transitions occur according to threshold- based rules: if the fraction of risk signals  $s / k_i \geq \theta_i$ , the node transitions to high arousal ( $H$ ); if  $s / k_i \leq \phi_i$ , it transitions to low arousal ( $L$ ); otherwise, it remains in medium arousal ( $M$ ). This individual- level implementation allows us to capture the heterogeneity in public responses and study transition dynamics at the microscopic scale.

Media nodes implement distinct feedback mechanisms that mirror our theoretical framework. Mainstream media nodes update their risk perception according to the stabilizing feedback rule  $p_{\text{risk}}^{\text{mainstream}}(t + 1) = (1 - \dot{X}_H + \dot{X}_L) / 2$ , where  $\dot{X}_H$  and  $\dot{X}_L$  represent the average high and low arousal proportions among their connected public nodes, while We media nodes

follow the amplifying feedback rule  $p_{\text{risk}}^{\text{wemedia}}(t + 1) = \dot{X}_H$ , directly amplifying high arousal states. The simulation enables detailed analysis of individual transition dynamics that cannot be captured by mean- field theory alone, tracking transition rates between emotional states for each node and computing transition matrices  $T_{ij}$  that describe the probability of transitioning from state  $i$  to state  $j$  within a given time window. This microscopic data reveals how transition dynamics depend on local network structure, individual threshold parameters, and global control parameters.

We validate the simulation against our mean- field theory by comparing macroscopic observables (global  $X_{H}$ ,  $X_{M}$ ,  $X_{L}$  proportions) with theoretical predictions across parameter space. The agreement between simulation and theory confirms the validity of our theoretical framework while the microscopic data provides additional insights into individual- level dynamics that inform our understanding of emotional bubble formation mechanisms.

# 746 Code and Data Availability

All code and data supporting the conclusions of this article are publicly available. The complete computational framework, including data preprocessing scripts, machine learning models, theoretical calculations, and network simulations, is accessible at https://github.com/wujlin/MFM. The repository contains detailed implementation of the mixed- feedback model (MFM), self- consistent equation solvers, phase transition analysis tools, and agent- based network simulation codes. Raw data from Weibo posts, processed datasets, and trained model parameters are available through the same repository. All analyses were conducted using Python 3.8 with standard scientific computing libraries (NumPy, SciPy, pandas, scikit- learn, NetworkX). The theoretical calculations utilize custom- developed numerical solvers for self- consistent equations and Jacobian matrix analysis. Computational requirements and detailed installation instructions are provided in the repository documentation to ensure full reproducibility of all results presented in this study.

# 759 Funding

This study was funded by the National Social Science Foundation of China (Grant No. 24AXW005, "Research on Telling Chinese National Stories from the Perspective of Strengthening the Chinese National Community Consciousness"). The funder played no role in study design, data collection, analysis and interpretation of data, or the writing of this manuscript.

# 765 Reference

1. Z. Liu, J. Wu, C. Y. H. Wu, X. Xia, Shifting sentiments: analyzing public reaction to COVID-19 containment policies in Wuhan and Shanghai through Weibo data. Humanit. Soc. Sci. Commun. 11, 1104

(2024). 2. F. Aslam, T. M. Awan, J. H. Syed, A. Kashif, M. Parveen, Sentiments and emotions evoked by news headlines of coronavirus disease (COVID-19) outbreak. Humanit. Soc. Sci. Commun. 7, 23 (2020). 3. W.-Y. S. Chou, A. and Budenz, Considering Emotion in COVID-19 Vaccine Communication: Addressing Vaccine Hesitancy and Fostering Vaccine Confidence. Health Commun. 35, 1718-1722 (2020). 4. R. E. Kasperson, et al., The Social Amplification of Risk: A Conceptual Framework. Risk Anal. 8, 177-187 (1988). 5. F. Jørgensen, A. Bor, M. S. Rasmussen, M. F. Lindholt, M. B. Petersen, Pandemic fatigue fueled political discontent during the COVID-19 pandemic. Proc. Natl. Acad. Sci. 119, e2201266119 (2022). 6. A. Goldenberg, E. Wesz, T. D. Sweeny, M. Cikara, J. J. Gross, The Crowd-Emotion-Amplification Effect. Psychol. Sci. 32, 437-450 (2021). 7. W. J. Brady, et al., Overperception of moral outrage in online social networks inflates beliefs about intergroup hostility. Nat. Hum. Behav. 7, 917-927 (2023). 8. A. D. I. Kramer, J. E. Guillory, J. T. Hancock, Experimental evidence of massive-scale emotional contagion through social networks. Proc. Natl. Acad. Sci. U. S. A. 111, 8788-8790 (2014). 9. R. Fan, J. Zhao, Y. Chen, K. Xu, Anger Is More Influential than Joy: Sentiment Correlation in Weibo. PLOS ONE 9, e110184 (2014). 10. M. J. Crockett, Moral outrage in the digital age. Nat. Hum. Behav. 1, 769-771 (2017). 11. J. Wang, et al., Global evidence of expressed sentiment alterations during the COVID-19 pandemic. Nat. Hum. Behav. 6, 349-358 (2022). 12. A. Hasell, Shared Emotion: The Social Amplification of Partisan News on Twitter. Digit. Journal. (2021). 13. M. Chong, M. and Choy, The Social Amplification of Haze-Related Risks on the Internet. Health Commun. 33, 14-21 (2018). 14. A. Giacomo, E. Calzola, G. Dimarco, Opinion Dynamics in Social Networks: Kinetic and Data-driven Modeling | SIAM. Soc. Ind. Appl. Math. (2025). Available at: https://www.siam.org/publications/siam-news/articles/opinion-dynamics-in-social-networks-kinetic-and-data-driven-modeling/ [Accessed 9 August 2025]. 15. J. Piao, J. Liu, F. Zhang, J. Su, Y. Li, Human-AI adaptive dynamics drives the emergence of information cocoons. Nat. Mach. Intell. 5, 1214-1224 (2023). 16. L. Helfmann, N. Djurdjevac Conrad, P. Lorenz-Spreen, C. Schutte, Modelling opinion dynamics under the impact of influencer and media strategies. Sci. Rep. 13, 19375 (2023). 17. R. K. Garrett, Echo chambers online?: Politically motivated selective exposure among Internet news users. J. Comput.-Mediat. Commun. 14, 265-285 (2009). 18. R. K. Garrett, Politically Motivated Reinforcement Seeking: Reframing the Selective Exposure Debate. J. Commun. 59, 676-699 (2009). 19. V. Pansanella, A. Srbu, J. Kertesz, G. Rossetti, Mass media impact on opinion evolution in biased digital environments: a bounded confidence model. Sci. Rep. 13, 14600 (2023). 20. C. A. Bail, et al., Exposure to opposing views on social media can increase political polarization. Proc. Natl. Acad. Sci. U. S. A. 115, 9216-9221 (2018). 21. N. J. Stroud, Polarization and Partisan Selective Exposure. J. Commun. 60, 556-576 (2010). 22. S. Zhou, X. Yang, Y. Wang, X. Zheng, Z. Zhang, Affective agenda dynamics on social media: interactions of emotional content posted by the public, government, and media during the COVID-19 pandemic. Humanit. Soc. Sci. Commun. 10, 797 (2023). 23. L. Lu, J. Xu, J. Wei, F. L. Shults, X. L. Feng, The role of emotion and social connection during the COVID-

19 pandemic phase transitions: a cross- cultural comparison of China and the United States. Humanit. Soc. Sci. Commun. 11, 237 (2024).24. O. Wilson, A. Flahault, China's U- turn in its COVID- 19 policy. Anaesth. Crit. Care Pain Med. 42, 101197 (2023).25. J. Heffner, M.- L. Vives, O. FeldmanHall, Emotional responses to prosocial messages increase willingness to self- isolate during the COVID- 19 pandemic. Personal. Individ. Differ. 170, 110020 (2021).26. M. P. Stijačić, K. Mišić, D. F. Durdević, Flattening the curve: COVID- 19 induced a decrease in arousal for positive and an increase in arousal for negative words. Appl. Psycholinguist. 44, 1069- 1089 (2023).27. D. R. Pereira, A. C. Teixeira- Santos, A. Sampaio, A. P. Pinheiro, Examining the effects of emotional valence and arousal on source memory: A meta- analysis of behavioral evidence. Emot. Wash. DC 23, 1740- 1763 (2023).28. P. L. Liu, COVID- 19 Information Seeking on Digital Media and Preventive Behaviors: The Mediation Role of Worry. Cyberpsychology Behav. Soc. Netw. 23, 677- 682 (2020).29. Y. Li, S. Luan, Y. Li, R. Hertwig, Changing emotions in the COVID- 19 pandemic: A four- wave longitudinal study in the United States and China. Soc. Sci. Med. 1982 285, 114222 (2021).30. N. Martinelli, et al., Time and Emotion During Lockdown and the Covid- 19 Epidemic: Determinants of Our Experience of Time? Front. Psychol. 11, 616169 (2020).31. M. Del Vicario, et al., The spreading of misinformation online. Proc. Natl. Acad. Sci. U. S. A. 113, 554- 559 (2016).32. M. Del Vicario, et al., Echo Chambers: Emotional Contagion and Group Polarization on Facebook. Sci. Rep. 6, 37825 (2016).33. Y. Wu, H. Xiao, F. Yang, Government information disclosure and citizen coproduction during COVID- 19 in China. Gov. Oxf. Engl. (2021). https://doi.org/10.1111/gove.12645.34. W. J. Brady, J. A. Willi, J. T. Jost, J. A. Tucker, J. J. Van Bavel, Emotion shapes the diffusion of moralized content in social networks. Proc. Natl. Acad. Sci. 114, 7313- 7318 (2017).35. S. Vosoughi, D. Roy, S. Aral, The spread of true and false news online. Science (2018). https://doi.org/10.1126/science.aap9559.36. A. Goldenberg, J. J. Gross, Digital Emotion Contagion. Trends Cogn. Sci. 24, 316- 328 (2020).37. S. Rathje, J. J. Van Bavel, S. van der Linden, Out- group animosity drives engagement on social media. Proc. Natl. Acad. Sci. U. S. A. 118, e2024292118 (2021).38. H. A. Schwartz, et al., Personality, Gender, and Age in the Language of Social Media: The Open- Vocabulary Approach. PLOS ONE 8, e73791 (2013).39. B. Pfefferbaum, C. S. Norris, Mental Health and the Covid- 19 Pandemic. N. Engl. J. Med. 383, 510- 512 (2020).40. H. Bisgin, N. Agarwal, X. Xu, A study of homophily on social media. World Wide Web 15, 213- 232 (2012).41. P. Dandekar, A. Goel, D. T. Lee, Biased assimilation, homophily, and the dynamics of polarization. Proc. Natl. Acad. Sci. 110, 5791- 5796 (2013).42. M. McPherson, L. Smith- Lovin, J. M. Cook, Birds of a Feather: Homophily in Social Networks. Annu. Rev. Sociol. 27, 415- 444 (2001).43. D. Centola, The Spread of Behavior in an Online Social Network Experiment. Science (2010). https://doi.org/10.1126/science.1185231.44. R. M. Bond, et al., A 61- million- person experiment in social influence and political mobilization. Nature 489, 295- 298 (2012).45. J. A. Russell, A circumplex model of affect. J. Pers. Soc. Psychol. 39, 1161- 1178 (1980).

46. V. Sanh, L. Debut, J. Chaumond, T. Wolf, DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. [Preprint] (2020). Available at: http://arxiv.org/abs/1910.01108 [Accessed 6 August 2025].

# Supplementary Information

# 1. Data Collection and Processing

# Dataset Overview

This study analyzes Long- COVID discourse on Sina Weibo during China's transition from strict COVID policy (December 2023). We compiled a comprehensive dataset of 294,434 posts selected through COVID- related hashtags (e.g., #Long- COVID#, #After recovering from COVID- 19#, #COVID后遗症#). The corpus comprises content from three distinct media types: We Media (49,288 posts,  $16.7\%$ ), mainstream media (36,480 posts,  $12.4\%$ ), and individual users (208,666 posts,  $70.9\%$ ), capturing the full spectrum of public discourse during this critical transition period.

# 88 Data Structure and Variables

Our integrated dataset contains three interlinked components:

User Profiles: Comprehensive demographic and account information including user ID, gender, location, verification status, follower count, following count, and account description. Key variables for node classification include verification status codes (mainstream media identified through official registry, We Media through specific verification codes, individual users through standard user identifiers) and profile descriptions containing industry identifiers.

Content Posts: Complete microblog entries with timestamp, content text, engagement metrics (forwards, comments, likes), media attachments, and source information. Temporal distribution analysis revealed significant discourse concentration in December 2023, coinciding with China's policy transition and increased public attention to Long- COVID symptoms.

Network Relations: User interaction patterns including forwarding relationships, comment networks, and mention structures that facilitate the propagation of emotional content across different media types.

# Data Processing Pipeline

We implemented a systematic preprocessing workflow to ensure data quality:

Content Filtering: Manual removal of irrelevant posts followed by automated filtering of entries with fewer than 10 characters, yielding 294,434 valid samples across the three media categories.

Text Normalization: Application of regular expressions to remove HTML elements (e.g., "//@xxx", "#hashtag"), translation of emoji icons into corresponding textual representations, and segmentation of Chinese characters using the jieba toolkit optimized for social media content.

Feature Engineering: Implementation of part- of- speech tagging, dependency parsing, and domain- specific stop word removal to optimize text for emotional and risk classification tasks.

Classification Framework: Development of specialized multi- class classifiers for:

- Emotional arousal classification (high/medium/low) achieving high overall accuracy- Risk signaling detection in mainstream media with superior precision- Risk signaling detection in We Media with competitive precision

# 2. Classification Model Performance

# Emotional Arousal Classification

Our emotion classification system integrates theoretical insights from psychological affect theory with advanced computational techniques, implemented through a multi- stage process that ensures both reliability and theoretical consistency.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-27/9ca076a7-360a-4b46-9317-b4e5b349dce7/eff612af779d4964438bd1bab6bca898696ce55c2581d8059206086672e82b5e.jpg)  
Figure S1. Classification model performance and data distribution. a, F1-weighted scores across three media types showing differential classification performance, with individual users achieving the highest classification accuracy, followed by mainstream media and We Media. b, Test recall distribution demonstrating model reliability with distinct patterns for each media type, showing characteristic Gaussian-like distributions with varying centers and spreads. c, Final test accuracy comparison across media types with confidence intervals. d, Coefficient of variation

analysis showing model stability, with individual users exhibiting the lowest variance, indicating the most consistent classification performance across validation folds.

# 909 Theoretical Foundation and Sample Selection

The classification framework builds upon Russell's circumplex model of affect, particularly its application to health crisis communication. Emotional content was categorized based on arousal levels:

High- arousal emotions: Include anger, fear, criticism, and sarcasm with terms such as "anti- science," "hypocritical," "unscrupulous," "intimidation," and "bias"- Medium- arousal emotions: Exist at the threshold between high and low arousal, with phrases like "don't underestimate," "don't panic," and "stop intimidating"- Low- arousal emotions: Cover uncertainty, avoidance, confusion, sadness, helplessness, and indifference

The confidence scoring mechanism employs stringent criteria for sample selection:

maximum single emotion category  $D_{i} = \left\{ \begin{array}{ll}\text{maximum} & \text{single emotion category}\\ \text{high} & \text{dominant category} > \text{others combined}\\ \text{moderate} & \text{multiple categories present} \end{array} \right.$

921

optimal moderate length range  $L_{i} = \left\{ \begin{array}{ll}\text{optimal} & \text{moderate length range}\\ \text{scaled down} & \text{short text}\\ \text{scaled down} & \text{long text} \end{array} \right.$

# 922 Annotation Protocol

The annotation process integrates theoretical rigor with practical feasibility through a structured protocol. Quality assurance measures include comprehensive annotator training, regular calibration sessions, and systematic dispute resolution procedures. The reliability metrics demonstrate strong consistency across all emotional categories:

-  $K_{\text{overall}}$  represents general inter-rater agreement across all arousal categories-  $K_{\text{high}}$  captures agreement on high-arousal emotional states-  $K_{\text{medium}}$  measures agreement on medium-arousal emotional states-  $K_{\text{low}}$  quantifies agreement on low-arousal emotional states

All  $K$  values demonstrate substantial agreement between annotators according to standard interpretation guidelines.

# 933 Model Architecture and Validation

The emotion classification model employs a carefully optimized DistilBERT architecture with domain- specific adaptations. The output layer design reflects the hierarchical nature of emotional arousal classification:

$$
P(y\vee x) = \mathrm{softmax}\left(W_{2}\mathrm{ReLU}\left(W_{1}h + b_{1}\right) + b_{2}\right)
$$

Comprehensive validation demonstrates robust performance across multiple metrics:

# Performance verification metrics:

Accuracy  $c_{v}$  represents cross- validation accuracy across independent test folds Precision  $h_{high}$  quantifies the model's ability to correctly identify high- arousal emotions  $\mathrm{F1}_{high}$  provides the harmonic mean of precision and recall for high- arousal classification

These metrics confirm the model's robust performance across different emotional categories and validation schemes.

This integrated approach ensures reliable emotion classification while maintaining theoretical consistency and practical applicability in health crisis communication contexts.

# Risk Signaling Classification

Risk signaling classification employs separate binary classifiers for mainstream media and We Media, recognizing the distinct communication patterns and institutional constraints of each media type.

Mainstream Media Risk Classification: Trained on professionally curated content with explicit risk terminology, achieving high precision through identification of formal risk indicators such as "health warning," "medical alert," and "potential complications."

We Media Risk Classification: Designed to capture informal risk expressions and personal testimonials, achieving competitive precision by recognizing colloquial risk language including "be careful," "concerning symptoms," and "worrisome trends."

# Simulation-Theory Validation Results

Figure S2 presents comprehensive validation of theoretical predictions using Poisson degree distribution networks, providing an independent validation of the mean- field theory framework.

Panel a shows the three- dimensional parameter space exploration with polarization field strength mapped across coupling parameter  $r$ , heterogeneity parameter  $\theta$ , and phase parameter  $\phi$ . The Poisson network simulations reproduce critical surface topology consistent with mean- field predictions.

Panel b validates theoretical predictions against simulation data across the  $\phi$  parameter range, showing excellent agreement between mean- field theory (blue) and network simulations (red) for

both high- arousal  $(X_H)$  and low- arousal  $(X_L)$  states. The close alignment confirms the accuracy of the self- consistent equation approach under Poisson degree distribution.

Panel c demonstrates model robustness across different network density assumptions (Base, Density, Max configurations), with consistently low RMSE values indicating stable performance regardless of specific network parameters.

Panel d presents polarization dynamics under systematic node removal, revealing system resilience with high polarization and low polarization states coexisting across various removal ratios. The filled regions indicate confidence intervals from ensemble averaging.

Panel e shows direct transition rate measurements confirming theoretical predictions of nonmonotonic transition behavior as a function of the  $\phi$  parameter.

Panel f illustrates the three- layer network architecture with information flow pathways: mainstream media (top layer), We Media (middle layer), and individual users (bottom layer), with arrows indicating bidirectional information exchange and feedback mechanisms.

# Power-Law Distribution (Scale-Free Networks)

Scale- free networks with degree distribution  $P(k) \sim k^{- \gamma}$  where  $\gamma$  represents the power- law exponent typical for social networks. The implementation uses truncated power- law distributions to maintain finite moments while preserving scale- free characteristics:

$$
P(k) = \frac{k^{-\gamma}}{\sum_{k = k_{min}}^{k_{max}}\sqcup k^{-\gamma}},k\in [k_{min},k_{max}]
$$

Critical analysis reveals:

Power- law network analysis:

- Critical point:  $r_c$  exhibits consistent values across different network realizations- Critical exponent:  $\gamma$  characterizes the correlation length divergence near criticality- Correlation length scaling:  $\xi \sim \vee r - r_c \dot{c}^{-\nu}$  with mean-field exponent

# Poisson Distribution (Random Networks)

Poisson degree distribution  $P(k) = \frac{\lambda^k e^{- \lambda}}{k!}$  represents homogeneous random connectivity with  $\lambda$  determined by network constraints:

$$
\lambda = \frac{k_{out,m}\cdot N_m + k_{out,w}\cdot N_w}{N_p}
$$

where  $k_{out,i}$  and  $N_{i}$  denote out- degrees and population sizes respectively.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-27/9ca076a7-360a-4b46-9317-b4e5b349dce7/155441478f15aaabf824903aaaa4805a4bda131d4865b8b5716570d8f23bf608.jpg)  
Figure S2. Network topology validation using Poisson degree distribution. The figure demonstrates that critical behavior emerges consistently under homogeneous random connectivity, supporting the universality of the phase transition phenomenon.

Critical behavior analysis reveals that both network topologies exhibit qualitatively similar phase transitions, with critical points occurring in comparable parameter ranges and consistent scaling behavior near the transition.

# 3. Parameter Specifications

The following parameters are utilized in our multi- scale analysis framework:

Table S1. Parameter definitions and specifications for the mixed-feedback model  

<table><tr><td>Parameter</td><td>Definition</td><td>Value Range</td><td>Calibration Method</td></tr><tr><td>r</td><td>Mainstream media removal ratio</td><td>[0, 1]</td><td>Systematic parameter scan</td></tr><tr><td>φ</td><td>Low arousal threshold</td><td>[0, 1]</td><td>Empirical data fitting</td></tr><tr><td>θ</td><td>High arousal threshold</td><td>[0, 1]</td><td>Empirical data fitting</td></tr></table>

<table><tr><td>XH</td><td>High arousal population fraction</td><td>[0, 1]</td><td>Self-consistent solution</td></tr><tr><td>XM</td><td>Medium arousal population fraction</td><td>[0, 1]</td><td>Self-consistent solution</td></tr><tr><td>XL</td><td>Low arousal population fraction</td><td>[0, 1]</td><td>Self-consistent solution</td></tr><tr><td>Pmainstream risk</td><td>Mainstream media risk perception</td><td>[0, 1]</td><td>Negative feedback formula</td></tr><tr><td>Pwemedia risk</td><td>We media risk perception</td><td>[0, 1]</td><td>Positive feedback formula</td></tr><tr><td>nm</td><td>Number of mainstream media sources</td><td>Integer</td><td>Empirical counting</td></tr><tr><td>nW</td><td>Number of We media sources</td><td>Integer</td><td>Empirical counting</td></tr><tr><td>ξ</td><td>Correlation length</td><td>Positive values</td><td>Jacobian eigenvalue analysis</td></tr><tr><td>v</td><td>Critical exponent</td><td>Positive values</td><td>Power-law fitting</td></tr><tr><td>rc</td><td>Critical removal ratio</td><td>[0, 1]</td><td>Phase transition analysis</td></tr></table>