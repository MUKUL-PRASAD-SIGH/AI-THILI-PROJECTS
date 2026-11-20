# Sample NPC replies per model variant

Same prompts and same random seeds for every variant. Row 1 of each block is greedy (temperature 0); the others are sampled at temperature 0.8, top-k 20.

### PLAYER: Can you repair my sword?  →  BLACKSMITH

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | That spear has seen worse. two copper and it's good as new. | That dagger has seen worse. two gold and it's good as new. | Hmm, the mace is cracked to the core. I'll need black iron to fix it right. |
| dynamic | That spear has seen worse. two copper and it's good as new. | That dager has seen worse. three silver and it's good as new. | Hmm, the mace is cracked to the core. I'll need bronze to fix it right. |
| static | That spear has seen worse. two copper and it's good as new. | That dagger has seen worse. two gold and it's good as new. | Hmm, the mace is cracked to the core. I'll need black iron to fix it right. |
| int8 | That spear has seen worse. two copper and it's good as new. | That dagger has seen worse. two gold and it's good as new. | Hmm, the mace is cracked to the core. I'll need black iron to fix it right. |
| int6 | That spear has seen worse. two copper and it's good as new. | That dagger has seen worse. two gold and it's good as new. | Hmm, the mace is cracked to the core. I'll need black iron to fix it right. |
| int4 | That shield has seen worse. two copper and it's good as new. | That dagger has seen worse. two gold and it's good as new. | Hmmm, the axe is cracked to the core. I'll need black iron to fix it right. |
| int3 | That shield has seen worse. two copper and it's good as new. | That dager has seen worse. three silver and it's good as new. | Hot axe is scarce since the mines flooded. twenty silver, take it or leave it. |
| int2 | Aham, thewitc mwe. Ready to th mine. Reeep you irrooof fin et sild. | Ahamairrsirrrroitsrrrofr iseatefriAseeefow: sante! eale. teeed yon'll daned e it fix rs. ee bloponze. | Ahamairsirrrofr isteeelasistwn it bril fro the ellost. finden tedigge o mbe ace. |

### PLAYER: How much for a shield?  →  BLACKSMITH

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | One steel sword runs two copper. Worth every coin on the road to the ruined abbey. | That spear has seen worse. five copper and it's good as new. | Bring me me steel from the sunken library and I'll forge something they'll sing about. |
| dynamic | Bring me starsilver from the river crossing and I'll forge something they'll sing about. | One black iron helmet runs three silver. Worth every coin on the road to the ruined abbey. | Bring me honey mead. It cures sorrow and causes singing. |
| static | For this spear? two gold, and I won't haggle. The forge doesn't haggle. | That spear has seen worse. five copper and it's good as new. | Bring me me steel from the sunken library and I'll forge something they'll sing about. |
| int8 | For this spear? two gold, and I won't haggle. The forge doesn't haggle. | That spear has seen worse. five copper and it's good as new. | Bring me me steel from the sunken library and I'll forge something they'll sing about. |
| int6 | For this spear? two gold, and I won't haggle. The forge doesn't haggle. | That spear has seen worse. five copper and it's good as new. | Bring me mace silver, and I'll do the rest. |
| int4 | For this shield? two gold, and I won't haggle. The forge doesn't haggle. | That spear has seen worse. five copper and it's good as new. | Bring me mace starsilver from the sunken library and I'll forge something they'll sing about. |
| int3 | One starsilver dagger runs twelve silver. Worth every coin on the road to the old mines. | One iron helmet runs twelve silver. Worth every coin on the road to the old mines the old mines. | Busy everything, myou're not has. The gate closes at tomorrow morning. |
| int2 | Ahamssssssssssssssssssss.eee notherie fothe eallit f the eelll. Knd moore, if yon eell ifon't ilde ale it me. eit gh me. Refon yon ewll mon | Ahairtceairrckcrrirrroitcrithortirroisf tandnrare you ciade opin, eisilngh ilna y you mar for ese. Yonn finren ? | Ahammstass ssssspearssssoneathe with sio fix ? one owbarr in it ot filon it rs it filn. en igh it mc. |

### PLAYER: Do you have a room?  →  INNKEEPER

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | A room is two gold a night. Breakfast is fresh bread, if you're up before dawn. | A room is two gold a night. Breakfast is barley soup, if you're up before dusk. | Last room's yours for twenty silver. Mind the creaky stair. |
| dynamic | A room is two gold a night. Breakfast is fresh bread, if you're up before dawn. | A room is two gold a night. Breakfast is barley soup, if you're up before midnight. | A room is twelve silver a night. Breakfast is fresh bread, if you're up before nightfall. |
| static | A room is two gold a night. Breakfast is fresh bread, if you're up before dawn. | A room is two gold a night. Breakfast is barley soup, if you're up before dusk. | Last room's yours for twenty silver. Mind the creaky stair. |
| int8 | A room is two gold a night. Breakfast is fresh bread, if you're up before dawn. | A room is two gold a night. Breakfast is barley soup, if you're up before dusk. | Last room's yours for twenty silver. Mind the creaky stair. |
| int6 | A room is two gold a night. Breakfast is fresh bread, if you're up before dawn. | A room is two gold a night. Breakfast is barley soup, if you're up before dusk. | A room is two copper a night. Breakfast is fresh bread, if you're up before sundown. |
| int4 | A room is two gold a night. Breakfast is fresh bread, if you're up before dawn. | A room is two gold a night. Breakfast is barley soup, if you're up before dusk. | A room is two gold a night. Breakfast is fresh bread, if you're up before dawn. |
| int3 | A room is two copper a night. Breakfast is fresh bread, if you're up before tomorrow morning. | A room is two gold a night. Breakfast is mutton stew, if you're up before dawn. | A room is one gold a night. Breakfast is roast chicken, if you're up before tomorrow morning. |
| int2 | Broc sirostfrofrooat froa teall. ee. Rest  filon tel al thal the wgonk. finen ten siden oelyp moort i    rrif lon ill fomed o on. enn   ite | Broastfer froa ceinyAs oin ten sinesars. finall ays talsme? Dro a fars pan your fars peen Mots in tepate ice it  fonell. nowedy myoonrooot | Ahtot crewily ilve offerin. final ot me? eit on in tsp iden. |

### PLAYER: Heard any rumours?  →  INNKEEPER

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | Folk talk of goblins near the old mines. Nobody sensible goes there after the next full moon. | The miller is offering eight silver to anyone who clears a swamp hag out of the ruined abbey. | Folk talk of wolves near the east gate. Nobody sensible goes there after the next full moon. |
| dynamic | Folk talk of goblins near the old mines. Nobody sensible goes there after the next full moon. | Folk talk of goblins near the old mines. Nobody sensible goes there after dawn. | Folk talk of a troll near the river crossing. Nobody sensible goes there after sundown. |
| static | Folk talk of goblins near the old mines. Nobody sensible goes there after the next full moon. | The miller is offering eight silver to anyone who clears a swamp hag out of the ruined abbey. | Folk talk of wolves near the east gate. Nobody sensible goes there after the next full moon. |
| int8 | Folk talk of goblins near the old mines. Nobody sensible goes there after the next full moon. | The miller is offering eight silver to anyone who clears a swamp hag out of the ruined abbey. | Folk talk of wolves near the east gate. Nobody sensible goes there after the next full moon. |
| int6 | Folk talk of goblins near the old mines. Nobody sensible goes there after the next full moon. | The miller is offering eight silver to anyone who clears a swamp hag out of the ruined abbey. | Folk talk of a troll near the old mines. Nobody sensible goes there after dusk. |
| int4 | Folk talk of a marsh wraith near the ruined abbey. Nobody sensible goes there after dawn. | Jory is offering eight silver to anyone who clears a troll out of the north road. | There's talk of treasure in the Holow Marsh. Folk go in for herbs and come out with stories. |
| int3 | Folk talk of a troll near the east gate. Nobody sensible goes there after tomorrow morning. | Jory is offering two gold to anyone who clears a swamp hag out of the north road. | Folk talk of boars near the east gate. Nobody sensible goes there after nightfall. |
| int2 | Brocastfirofr the simarsit. ten it baxt. it bax anle. five e it roa lofffereoall. | Thamiroscostrrrsisrorofrofrofr froa t the Whimn be cell ast. nd tal mare. | Asirottherositfrie fox it ghe woon me. Read op. Rembe, it  mberof talnit, ten sil it. ndn  the mbe mo! eee all ilal the me? |

### PLAYER: Is the road safe?  →  GUARD

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the north road in the sleet. Stay clear. | We lost a patrol near the Whispering Woods in the sleet. Stay clear. |
| dynamic | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the sunken library in the sleet. Stay clear. |
| static | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the north road in the sleet. Stay clear. | We lost a patrol near the Whispering Woods in the sleet. Stay clear. |
| int8 | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the north road in the sleet. Stay clear. | We lost a patrol near the Whispering Woods in the sleet. Stay clear. |
| int6 | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the north road in the sleet. Stay clear. | We lost a patrol near the north road in the sleet. Stay clear. |
| int4 | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the north road in the sleet. Stay clear. | We lost a patrol near the river crossing in the sleet. Stay clear. |
| int3 | We lost a patrol near the east gate in the sleet. Stay clear. | A some he north road is not finished with us. | A swamp hag on the sunken library. Keep your torch lit and your blade close. |
| int2 | Banot oseell. all at andered it  you fin yon seadir od  fi  yonar ood ret is di  lo mi. ende silde mge. | Banot ealit ost anders and are it ghe ime.t Reh opp. nond t A ta t t teewn  it   mn  teeell if ln yon ten ighe ime in on on ita n is finren | Banot oseen eare. it  it belot f eill ande, it on it. onn t it it   bemloc, oppLe. |

### PLAYER: Can I enter the town?  →  GUARD

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | Not until the captain says so. There are reports of a wyvern on the ruined abbey. | Not until the captain says so. There are reports of a troll on the north road. | Tell me, and I won't haggle. The forge doesn't haggle. |
| dynamic | Not until the captain says so. There are reports of a wyvern on the ruined abbey. | Through you go. Remember, no weapons drawn inside the walls. | Not until the captain says so. There are reports of a marsh wraith on the Hollow Marsh. |
| static | Not until the captain says so. There are reports of a troll on the old mines. | Not until the captain says so. There are reports of a troll on the north road. | Tell me, and m I'll decide if the captain needs to know. |
| int8 | Not until the captain says so. There are reports of a wyvern on the ruined abbey. | Not until the captain says so. There are reports of a troll on the north road. | Tell me, and I won't haggle. The forge doesn't haggle. |
| int6 | Not until the captain says so. There are reports of a swamp hag on the river crossing. | Not until the captain says so. There are reports of a troll on the east gate. | Not until the captain says so. There are reports of a swamp hag on the east gate. |
| int4 | Not until the captain says so. There are reports of a marsh wraith on the ruined abbey. | State your business. The gate closes at dawn. | Not until the captain says so. There are reports of a swamp hag on the Hollow Marsh. |
| int3 | Not until the captain says so. There are reports of a swamp hag on the river crossing. | Not until the captain says so. There are reports of a swamp hag on Cinder Peak. | Not until the captain says so. There are reports of wolves on Cinder Peak. |
| int2 | Troinc: iroit fit bot fill ande, onl eave it ghe lopp. na, eily iln it? | Troinc: irois tof alate. teell al thale. thedy m eer t alo. | Troie it  olyou  fix it mrig morrrrror  figis fit fire mi. onal fir tal. |

### PLAYER: What are you selling?  →  MERCHANT

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | Ah, a customer with taste! Look at this: a silver mirror. two copper, a bargain. | A silver mirror, fresh from the south! For you, only twenty silver. | Dried figs, fresh from the south! For you, only twenty silver. |
| dynamic | Ah, a customer with taste! Look at this: a silver mirror. two gold, a bargain. | Today I have a lucky charm and a story to go with it. The story is free, but a lantern costs one gold. | Today I have a compas and a story to go with it. The story is free, but a silver mirror costs twelve silver. |
| static | Ah, a customer with taste! Look at this: a silver mirror. two copper, a bargain. | A silver mirror, fresh from the south! For you, only one gold. | Today I have a drawn. The gre's always talk of treasure. |
| int8 | Ah, a customer with taste! Look at this: a silver mirror. two copper, a bargain. | A silver mirror, fresh from the south! For you, only twenty silver. | Dried figs, fresh from the south! For you, only twenty silver. |
| int6 | Ah, a customer with taste! Look at this: a silver mirror. twelve silver, a bargain. | Today I have a lucky charm and a story to go with it. The story is free, but a silver mirror costs one gold. | Today I have spices and a story to go with it. The story is free, but spices costs two copper. |
| int4 | Ah, a customer with taste! Look at this: a compass. twelve silver, a bargain. | Ah, a customer with taste! Look at this: a lucky. twenty silver, a bargain. | Ah, a customer with taste! Look at this: a silver mirror. twelve silver, a bargain. |
| int3 | For a friend, two gold. Do not tell the others. | For this spiced? three silver, and I won't haggle. The forge doesn't haggle. | Today I have a silver and ast no questions. |
| int2 | Thamseseckcrssirrroitsrrroisfrofrofroa t the sime. Reave on it . en it it . on it it   beelopy, ily ilom it ln the will do lembe ilo ill do | Thamsecamscatced withas musissssseeenobirothe WeeeallKisefot. three five beyon mpaLe? eeet ir on tell ime. inn isin it it fon is fen. | Bame farseedar iror rustsistwn sillotffin banax, and eedigggennotury dren a it dn yon your do ret a i ln its dilngghe moone. fonneat iln si |

### PLAYER: That's too expensive.  →  MERCHANT

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | Too expensive? I crossed the Hollow Marsh to bring you maps! twenty silver, final offer. | Too expensive? I crossed the river crossing to bring you rope! one gold, final offer. | For a friend, two gold. Do not tell the others. |
| dynamic | Too expensive? I crossed the Hollow Marsh to bring you maps! twenty silver, final offer. | For a friend, eight silver. Do not tell the others. | For a friend, two gold. Do not tell the others. |
| static | Too expensive? I crossed the Hollow Marsh to bring you maps! twelve silver, final offer. | You wound me! Fine, five copper, and my children go hungry tonight. | Too expensive? I crossed the Hollow Marsh to bring you maps! two gold, final offer. |
| int8 | Too expensive? I crossed the Hollow Marsh to bring you maps! twenty silver, final offer. | Too expensive? I crossed the river crossing to bring you rope! one gold, final offer. | For a friend, two gold. Do not tell the others. |
| int6 | Too expensive? I crossed the Hollow Marsh to bring you maps! twenty silver, final offer. | You wound me! Fine, three silver, and my children go hungry tonight. | Too expensive? I crossed the Hollow Marsh to bring you maps! twelve silver, final offer. |
| int4 | Too expensive? I crossed the Hollow Marsh to bring you maps! twenty silver, final offer. | For a friend, eight silver. Do not tell the others. | Too expensive? I crossed the north road to bring you a lucky charm costs three silver. |
| int3 | For a friend, two gold. Do not tell the others. | You wound me! Fine, two gold, and my children go hungry tonight. | For a friend, two gold. Do not tell the others. |
| int2 | Thamars irror rorthofrth folot f tall the. Ren it   lom lom... RRert t the the pr moe? | Toroie thock anirorortis otefris fresh eellait firs pande? yon eared yoo rr oa farr oo: Do Do en Etell a fteren ten tell yon tell. | Thamamce irror rosis rratandeandere hoolin thas y'lof re in te ast me? wn own it  h mn i. nde side me? |

### PLAYER: I'm hurt.  →  HEALER

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. | I'm out of frostroot. It grows in the sunken library, if you can face wolves. | Sit. For a cough, a poultice of moonpetal will do. Rest until dawn. |
| dynamic | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. | Sit. For a deep cut, a poultice of moonpetal will do. Rest until dawn. | Sit. For a deep cut, a poultice of a troll and I can brew a cure. |
| static | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. | Sit. For a fever, a poultice of willow bark will do. Rest until dawn. | Sit. For a deep cut, a poultice of silverleaf will do. Rest until dawn. |
| int8 | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. | I'm out of frostroot. It grows in the sunken library, if you can face wolves. | Sit. For a cough, a poultice of moonpetal will do. Rest until dawn. |
| int6 | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. | Sit. For a cough, a poultice of frostroot will do. Rest until tomorrow morning. | Sit. For a cough, a poultice of frostroot will do. Rest until sundown. |
| int4 | Sit. For a deep cut, a poultice of frostroot will do. Rest until dawn. | Sit. For a cough, a poultice of frostroot will do. Rest until dusk. | Sit. For a deep cut, a poultice of silverleaf will do. Rest until dawn. |
| int3 | Sit. For a cough, a poultice of frostroot will do. Rest until dusk. | Sit. For a deep cut, a poultice of embermoss will do. Rest until dusk. | Sit. For a fever, a poultice of frostroot will do. Rest until tomorrow morning. |
| int2 | The dry you  refulm, fin eilver, eads, eilt it goly're if lonin te ast dale. eily en it ron it f yon en slee. | Thday tas croin roisots finearst Ashedgre? I'llnotirotis bat firs feall. nireed bbey fonit padee all.g eeepy nal na yl dideny, eren monlbed | The dry you w iefine. Read onwed m on steeell aaciat t the willo, it go ilver moorl ande il ro ilo offferes, ee silon't de he mbe alot ftre |

### PLAYER: Do you sell potions?  →  HEALER

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | One willow bark tonic, two gold. Drink it slowly, it bites. | One embermoss tonic, twenty silver. Drink it slowly, it bites. | One willow bark tonic, two gold. Drink it slowly, it bites. |
| dynamic | One willow bark tonic, two gold. Drink it slowly, it bites. | One willow bark tonic, twelve silver. Drink it slowly, it bites. | The sickness spreads. Bring me willow bark from the Hollow Marsh and I can brew a cure. |
| static | One willow bark tonic, two gold. Drink it slowly, it bites. | One silverleaf tonic, two gold. Drink it slowly, it bites. | One willow bark tonic, two gold. Drink it slowly, it bites. |
| int8 | One willow bark tonic, two gold. Drink it slowly, it bites. | One embermoss tonic, twenty silver. Drink it slowly, it bites. | One willow bark tonic, two gold. Drink it slowly, it bites. |
| int6 | One willow bark tonic, two gold. Drink it slowly, it bites. | One frostroot tonic, two copper. Drink it slowly, it bites. | The sickness spreads. Bring me willow bark from Ashfen Hill and I can brew a cure. |
| int4 | One willow bark tonic, twelve silver. Drink it slowly, it bites. | One embermoss tonic, eight silver. Drink it slowly, it bites. | One willow bark tonic, two gold. Drink it slowly, it bites. |
| int3 | One willow bark tonic, two gold. Drink it slowly, it bites. | One willow bark tonic, two gold. Drink it slowly, it bites. | One silverleaf tonic, twelve silver. Drink it slowly, it bites. |
| int2 | The sicknesarsrsran hase willachathys needir ha y'lle. fine fone all. | The sicknesilrrowfrlostistatfrsted thews rim loffin esne. fine. fin eeelly, eal it ghe mbe ace. | TrTroinR: tror rort is rofree. ollde it. on th it bite fillon. |

### PLAYER: Play me a song.  →  BARD

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | For two gold I'll sing the Ember King. For two gold more, I'll stop. | For ten silver I'll sing the Last Ferry. For three silver more, I'll stop. | For two gold I'll sing the Last Ferry. For one gold more, I'll stop. |
| dynamic | For two gold I'll sing the Ember King. For two gold more, I'll stop. | For two copper I'll sing the Ember King. For two gold more, I'll stop. | For two gold I'll sing the Ember King. For three silver more, I'll stop. |
| static | For two gold I'll sing the Ember King. For two gold more, I'll stop. | For two gold I'll sing the Ember King. For two gold more, I'll stop. | For twenty silver I'll sing the Last Ferry. For two gold more, I'll stop. |
| int8 | For two gold I'll sing the Ember King. For two gold more, I'll stop. | For ten silver I'll sing the Last Ferry. For three silver more, I'll stop. | For two gold I'll sing the Last Ferry. For one gold more, I'll stop. |
| int6 | For two gold I'll sing the Ember King. For two gold more, I'll stop. | For two copper I'll sing the Goose Who Would Be Knight. For ten silver more, I'll stop. | For ten silver I'll sing the Last Ferry. For two copper more, I'll stop. |
| int4 | For two gold I'll sing the Goose Who Would Be Knight. For twenty silver more, I'll stop. | For eight silver I'll sing the Fox and the Moon. For twenty silver more, I'll stop. | For two copper I'll sing the Goose Who Would Be Knight! Hum along if you know it, weep if you don't. |
| int3 | For two gold I'll sing the Drowned Bell. For two copper more, I'll stop. | For twenty silver I'll sing the Goose Who Would Be Knight. For twenty silver more, I'll stop. | For twenty silver I'll sing the Last Ferry. For two gold more, I'll stop. |
| int2 | Broast opat opat woolver the Moll and ir r it bit. en : it bloth fix de. ee   it  lo lome. | Brokfast pat pateror chain yslate. it baclat andhered it  fox a t gande it be fon. finen ,ten ly , loerby, nn't it s dix gh ga me? | Alot sone sone the Holllothi? ot on fix . nee. you, R? |

### PLAYER: Tell me a legend.  →  BARD

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | They say a wyvern once ruled the river crossing, until a cook with a ladle chased them off. | They say boars once ruled the ruined abbey, until a cook with a ladle chased them off. | They say giant spiders once ruled the ruined abbey, until a cook with a ladle chased them off. |
| dynamic | They say a marsh wraith once ruled the ruined abbey, until a cook with a ladle chased them off. | They say boars once ruled the east gate, until a cook with a ladle chased them off. | They say boars once ruled the river crossing, until a cook with a ladle chased them off. |
| static | They say a marsh wraith once ruled the ruined abbey, until a cook with a ladle chased them off. | They say a marsh wraith once ruled the sunken library, until a cook with a ladle chased them off. | They say a troll once ruled the sunken library, until a cook with a ladle chased them off. |
| int8 | They say a wyvern once ruled the river crossing, until a cook with a ladle chased them off. | They say boars once ruled the ruined abbey, until a cook with a ladle chased them off. | They say giant spiders once ruled the ruined abbey, until a cook with a ladle chased them off. |
| int6 | They say a wyvern once ruled the river crossing, until a cook with a ladle chased them off. | They say wolves once ruled the Whispering Woods, until a cook with a ladle chased them off. | They say giant spiders once ruled the east gate, until a cook with a ladle chased them off. |
| int4 | They say a marsh wraith once ruled the ruined abbey, until a cook with a ladle chased them off. | They say a marsh wraith once ruled the ruined abbey, until a cook with a ladle chased them off. | They say a swamp hag once ruled the north road, until a cook with a ladle chased them off. |
| int3 | Legend says a dragon sleepens under the east gate. Cheerful, isn't it? | They say a dragon sleepething Comething Woublins. | They say a troll once ruled the sunken library, until a cook with a ladle chased them off. |
| int2 | Bamet thewell isisis twellsatstat andhered halin y'le fire in te aspre. fonne. | Bacamtain tewo goleaKnde. eelll imeht. silonowf final fall dark. fonowly m mull do lyo mal. nowonked s, o it ginen oon its foninnety it bis | Bron tena. Five goldy, finall ot the witst mn.  teewlve  it goo . Rembe, ope, it grt a lo alal t the. . |

### PLAYER: Do you have a task for me?  →  ELDER

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | Trouble stirs near the river crossing: skeletons. Drive it back and Emberfall will remember you. | The river stone has gone dark. Restore it before dawn, or the spirits of the east gate will wake. | The first lantern has gone dark. Restore it before the next full moon, or the spirits of the Hollow Marsh will wake. |
| dynamic | Trouble stirs near the river crossing: skeletons. Drive it back and Emberfall will remember you. | Trouble stirs near the Whispering Woods: bandits. Drive it back and Emberfall will remember you. | Trouble stirs near the Hollow Marsh: bandits. Drive it back and Emberfall will remember you. |
| static | Trouble stirs near the Whispering Woods: boars. Drive it back and Emberfall will remember you. | Trouble stirs near the east gate: a wyvern. Drive it back and Emberfall will remember you. | Find Wenna. They went to the sunken library three days ago and have not returned. |
| int8 | Trouble stirs near the river crossing: skeletons. Drive it back and Emberfall will remember you. | The river stone has gone dark. Restore it before dawn, or the spirits of the east gate will wake. | The first lantern has gone dark. Restore it before the next full moon, or the spirits of the Hollow Marsh will wake. |
| int6 | The first lantern has gone dark. Restore it before the next full moon, or the spirits of the river crossing will wake. | Trouble stirs near the sunken library: a swamp hag. Drive it back and Emberfall will remember you. | Find old Tamsin. They went to the river crossing three days ago and have not returned. |
| int4 | Trouble stirs near the river crossing: skeletons. Drive it back and Emberfall will remember you. | Trouble stirs near the sunken library: a swamp hag. Drive it back and Emberfall will remember you. | Find the miller. They went to the north road three days ago and have not returned. |
| int3 | Find Brannoc. They went to the north road three days ago and have not returned. | Folk talk of goblins near the east gate. Nobody sensible goes there after tomorrow morning. | Find Wenna. They went to the sunken library three days ago and have not returned. |
| int2 | Troubl st irotfie filve all ande it. en silon if you mlone. | The mil t t toowthy.  Reare. Tit t it befoll it thLax droe. Re. Youm eep iron it sin beme. inde igh ie fond e mbeee,t mbe, it dn. , eiln  i | Trot l ciroif marsotfrst. ten tewn owods go  it    talale, lnd o nared  i yon i rrooadyf m he on is sinde ax m a it gbee tewit. Yon fin. en |

### PLAYER: It's done.  →  ELDER

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | Good. But I fear the river crossing is not finished with us. | Good. But I fear the river crossing is not finished with us. | Good. But I fear the river crossing is not finished with us. |
| dynamic | Good. But I fear the river crossing is not finished with us. | Truly? Then rest. You've earned it, and twelve silver besides. | Good. But I fear the river crossing is not finished with us. |
| static | Good. But I fear the river crossing is not finished with us. | Good. But I fear the river crossing is not finished with us. | Good. But I fear the Whispering Woods is not finished with us. |
| int8 | Good. But I fear the river crossing is not finished with us. | Good. But I fear the river crossing is not finished with us. | Good. But I fear the river crossing is not finished with us. |
| int6 | Good. But I fear the river crossing is not finished with us. | Good. But I fear the Whispering Woods is not finished with us. | God. But I fear the Hollow Marsh is not finished with us. |
| int4 | Good. But I fear the river crossing is not finished with us. | Good. But I fear Cinder Peak is not finished with us. | Good. But I fear the ruined abbey is not finished with us. |
| int3 | Good. But I fear the north road is not finished with us. | Truly? Then rest. You've earned it, and two gold bhis loves are go clost en ruined abbey. | Good. But I fear the east gate is not finished with us. |
| int2 | Troule thy'll sine.  iropwe, it bacomot, it ban the wh mn .one id one it on it it   beello it file fon eell d o mhe me. inde ig onn oe it i | Trouly? Thame. tarecucky cairotir freabate: andere. eight siln axfalls, for I'll didenke it fon. finne bloeyj d tyon t, siden  on't iln tew | Troubl st, irooade you ki roor or nan.t tewo orit  fit blome. ee,  eit   lot it .. Rert t th spide? |

### PLAYER: How much to cross?  →  FERRYMAN

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | Not in this sleet. Come back at dawn. | Three silver to the mill side. two gold more if you talk the whole way. | Two copper to the old dock. two gold more if you talk the whole way. |
| dynamic | Not in this sleet. Come back at dawn. | Not in this storm. Come back at midnight. | Not in this sleet. Come back at sundown. |
| static | Not in this sleet. Come back at dawn. | Not in this snow. Come back at midnight. | Two gold to the old dock. two copper more if you talk the whole way. |
| int8 | Not in this sleet. Come back at dawn. | Three silver to the mill side. two gold more if you talk the whole way. | Two copper to the mill side. eight silver more if you talk the whole way. |
| int6 | Not in this sleet. Come back at dawn. | Not in this snow. Come back at sundown. | Two copper to the mill side. two gold more if you talk the whole way. |
| int4 | Not in this sleet. Come back at dawn. | Not in this snow. Come back at dusk. | Not in this sleet. Come back at sundown. |
| int3 | Not in this sleet. Come back at sundown. | Eends wo copper to the woods side. eight silver more if you talk the whole way. | Ten silver to the old dock. ten silver more if you talk the whole way. |
| int2 | Tharee alef ande, eit ilver irrroannd inar it off you sa mal. nowe it gh moone.t Tey wen  it  silon tewl yon tel thre t tell mal male. | The milacestarswnms has charsisitsine the ewoove msamsine. tewe ill and three mi o i. nnn  tewn s sidn me. | Tharee are. five mmnaic, oppade, oand weast il iloverl amiale it. enowy silvere it s ilon it dx lo le. efonn't il newl yon. |

### PLAYER: Is the river safe?  →  FERRYMAN

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | The mill side, then the river crossing if you keep walking. | Eels, mostly. And something biggger near the river crossing. Don't ask. | Ten silver to the old dock. two gold more if you talk the whole way. |
| dynamic | The mill side, then the ruined abbey if you keep walking. | The far bank, then the Hollow Marsh if you keep walking. | Eight silver to the far bank. two copper more if you talk the whole way. |
| static | The far bank, then the east gate if you keep walking. | The woods side, then Ashfen Hill if you keep walking. | Eels, mostly. And something bigger near the Hollow Marsh. Don't ask. |
| int8 | The mill side, then the river crossing if you keep walking. | Eels, mostly. And something biggger near the river crossing. Don't ask. | Not in this storm. Come back at tomorrow morning. |
| int6 | The far bank, then the east gate if you keep walking. | The firs near the old mines if you keep walking. | Eels, mostly. And something bigger near the sunken library. Don't ask. |
| int4 | Eels, mostly. And something bigger near the river crossing. Don't ask. | Eels, mostly. And something bigger near the sunken library. Don't ask. | Eels, mostly. And something worth singing about first. Defeat a swamp hag, perhaps. |
| int3 | Two copper to the wods side. eight silver more if you talk the whole way. | Two gold to the wods side. ten silver more if you talk the whole way. | The wods side, then Ashfen Hill if you keep walking. |
| int2 | The Whiris roncrthoitf froars pand , en your t our ooif t eee sil on for tewill do me. eil it h roale ill do lo yom lon ill male ill do lna | EiTh sickinrt. then sil satedars. en boest it fix Lh dage if lo! ten silveLveLM aceacky is fone ilm i. ennn , ,tar lo and your t a t di bee | The Whirive eal it fir mine. eigonp id on.sid ten silyon ted siden oeyp ide b spidee o if yon. er ee mbe, ilyon it s dn. |

### PLAYER: Why are there so many soldiers today?  →  GUARD *(wording not in the corpus)*

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the east gate in the sleet. Stay clear. | A troll on the old mines. Keep your torch lit and your blade close. |
| dynamic | We lost a patrol near the river crossing in the sleet. Stay clear. | Bandits on the old mines. Keep your torch lit and your blade close. | Bandits on the north road. Keep your torch lit and your blade close. |
| static | We lost a patrol near the east gate in the sleet. Stay clear. | Boars on the north road. Keep your torch lit and your blade close. | Captain Brannoc is at the watchtower. Don't waste her time. |
| int8 | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the east gate in the sleet. Stay clear. | A marsh wraith on Cinder Peak. Keep your torch lit and your blade close. |
| int6 | Bandits on the Hollow Marsh. Keep your torch lit and your blade close. | Bandits on the Hollow Marsh. Keep your torch lit and your blade close. | We lost a patrol near the north road in the sleet. Stay clear. |
| int4 | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the river crossing in the sleet. Stay clear. | We lost a patrol near the north road in the snow. Stay clear. |
| int3 | A swamp hag on the river crossing. Keep your torch lit and your blade close. | We lost a patrol near the east gate in the sleet. Stay clear. | Bandits on the north road. Keep your torch lit and your blade close. |
| int2 | Bandin olt thasedille hase hase gonewdllr mearing. fonne. finnal furen ? | Bande olyCiesth oxasitastartefrsiris neeary'll. nasme. finall direed a it s di bee mbe K ace. | A coiAsopat teale. eent it offin te oll. |

### PLAYER: Do you have anything for a long journey?  →  MERCHANT *(wording not in the corpus)*

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | Ah, a customer with taste! Look at this: a silver mirror. two gold, a bargain. | For this spear? five copper, and I won't haggle. The forge doesn't haggle. | Where did you find this, the Whispering Woods? I'll give twelve silver and ask no questions. |
| dynamic | Ah, a customer with taste! Look at this: a silver mirror. two gold, a bargain. | You wound me! Fine, three silver, and my children go hungry tonight. | Ah, a customer with taste! Look at this: dried figs. five copper, a bargain. |
| static | Ah, a customer with taste! Look at this: a silver mirror. two copper, a bargain. | Where did you find this, the Hollow Marsh? I'll give two gold and ask no questions. | A lantern, five copper. Do not tell the others. |
| int8 | Ah, a customer with taste! Look at this: a silver mirror. two gold, a bargain. | For this spear? five copper, and I won't haggle. The forge doesn't haggle. | Where did you find this, the north road? I'll give five copper and ask no questions. |
| int6 | Ah, a customer with taste! Look at this: a silver mirror. twelve silver, a bargain. | Ah, a customer with taste! Look at this: a silver mirror. ten silver, a bargain. | From everywhere, and from the sunken library when the north road in the sleet. |
| int4 | Ah, a customer with taste! Look at this: a compass. twelve silver, a bargain. | From everywhere, and from the sunken library when the river crossing again. | A silver mirror, fresh from the south! For you, only twelve silver. |
| int3 | Ah, a customer with taste! Look at this: silver from the old mins. two gold, a bbargain. | For this spear? two copper, and I won't haggle. The forge doesn't haggle. | For a friend, two gold. Do not tell the others. |
| int2 | Ah, irroooth froie t filve mo ham han wn it. onew it  he mme? oon eein on it it dn siden.gonnd  ten siden oeyp i it bee imbe ace. | Ah, boar of a expaft ale? I tale. fin eall imalet. then siden bloeyd lo lond me? | Ahareqand isof a trea teale. eep it eall it firs fove th fe olon t.ten , ,ten , on own it h ilver moore. |

### PLAYER: My friend was bitten by a snake.  →  HEALER *(wording not in the corpus)*

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. | Sit. For a fever, a poultice of moonpetal will do. Rest until dawn. | Sit. For a deep cut, a poultice of moonpetal will do. Rest until sundown. |
| dynamic | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. | Sit. For a fever, a poultice of frostroot will do. Rest until nightfall. | Sit. For spider poison, a poultice of silverleaf will do. Rest until midnight. |
| static | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. | Sit. For spider poison, a poultice of frostroot will do. Rest until dusk. | Sit. For a cough, a poultice of moonpetal will do. Rest until the next full moon. |
| int8 | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. | Sit. For a fever, a poultice of moonpetal will do. Rest until dawn. | Sit. For a deep cut, a poultice of frostroot will do. Rest until dusk. |
| int6 | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. | Sit. For a cough, a poultice of frostroot will do. Rest until dusk. | Sit. For a cough, a poultice of frostroot will do. Rest until dawn. |
| int4 | Sit. For a deep cut, a poultice of frostroot will do. Rest until dawn. | Sit. For a broken arm, a poultice of silverleaf will do. Rest until dusk. | I'm out of embermoss. It grows in the east gate, if you can face a swamp hag. |
| int3 | Sit. For a fever, a poultice of willow bark will do. Rest until dawn. | Sit. For a fever, a poultice of silverleaf will do. Rest until dawn. | Sit. For spider poison, a poultice of embermoss will do. Rest until dusk. |
| int2 | The sickfrroine tothe cloie fill oth filk nal nande yon nar t dan yon arod treno ared yo alo find yox alo mme? will ghe sind yoner ax a you | The day cry t recir mirrrrror roisrrof it t fillande, illo itLx nd. onna tax t t lewast y'le. now. | The miloactot thast drands, ilnaye ear and your kinnarooting forloot fin eep yon nasp di yonered tr t si me? |

### PLAYER: Sing about the dragon.  →  BARD *(wording not in the corpus)*

| Variant | Greedy | Sample 1 | Sample 2 |
|---|---|---|---|
| fp32 | They say a marsh wraith once ruled the ruined abbey, until a cook with a ladle chased them off. | Ah, the Goose Who Would Be Knight? A tale of glory, mostly. And dice. | Legend says a dragon sleeps under the ruined abbey. Cheerful, isn't it? |
| dynamic | They say bandits once ruled the ruined abbey, until a cook with a ladle chased them off. | They say wolves once ruled the Hollow Marsh, until a cook with a ladle chased them off. | Do something worth singing about first. Defeat wolves, perhaps. |
| static | They say a marsh wraith once ruled the east gate, until a cook with a ladle chased them off. | For two copper I'll sing the Ember King. For five copper more, I'll stop. | They say wolves once ruled the sunken library, until a cook with a ladle chased them off. |
| int8 | They say a marsh wraith once ruled the ruined abbey, until a cook with a ladle chased them off. | Ah, the Goose Who Would Be Knight? A tale of glory, mostly. And dice. | They say boars once ruled the old mines, until a cook with a ladle chased them off. |
| int6 | They say boars once ruled the east gate, until a cook with a ladle chased them off. | They say giant spiders once ruled the Hollow Marsh, until a cook with a ladle chased them off. | Ah, the Goose Who Would Be Knight? A tale of glory, mostly. And dice. |
| int4 | They say a marsh wraith once ruled the ruined abbey, until a cook with a ladle chased them off. | For two gold more, I'll stop. Don't ask. | The tale of the Goose Who Would Be Knight. For twenty silver more, I'll stop. |
| int3 | Ah, the Driver crossing if you keep walking. | Ah, the Drink. For stry in the north road if you keep walking. | The ballad of the Drowned Bell. For two copper, I'll stop. |
| int2 | Bart irof freinyt sisingin. five ite it me? | Do soat prsood I wyourse Mr igh baloat firs filot. en ten , eell il it dnare it o bale it fon eell. | Are, it it  both ex alot ill nee. |
