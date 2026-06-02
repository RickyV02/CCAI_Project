// ==========================================
// INIT KNOWLEDGE GRAPH
// ==========================================

// ==========================================
// 1. CONSTRAINTS
// ==========================================
CREATE CONSTRAINT game_name IF NOT EXISTS FOR (g:Game) REQUIRE g.name IS UNIQUE;
CREATE CONSTRAINT boss_name IF NOT EXISTS FOR (b:Boss) REQUIRE b.name IS UNIQUE;
CREATE CONSTRAINT mechanic_name IF NOT EXISTS FOR (m:Mechanic) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;
CREATE CONSTRAINT studio_name IF NOT EXISTS FOR (s:Studio) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT platform_name IF NOT EXISTS FOR (p:Platform) REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT blogpost_title IF NOT EXISTS FOR (b:BlogPost) REQUIRE b.title IS UNIQUE;
CREATE CONSTRAINT source_url IF NOT EXISTS FOR (s:Source) REQUIRE s.url IS UNIQUE;

// ==========================================
// 2. NODI BASE: GENRES, STUDIOS, PLATFORMS, MECHANICS
// ==========================================
// Generi
MERGE (:Genre {name: "Soulslike"});
MERGE (:Genre {name: "Action RPG"});
MERGE (:Genre {name: "Action"});
MERGE (:Genre {name: "Open World"});
MERGE (:Genre {name: "Roguelike"});
MERGE (:Genre {name: "Survival Horror"});
MERGE (:Genre {name: "Metroidvania"});

// Studios
MERGE (:Studio {name: "FromSoftware"});
MERGE (:Studio {name: "CD Projekt Red"});
MERGE (:Studio {name: "Capcom"});
MERGE (:Studio {name: "Larian Studios"});
MERGE (:Studio {name: "Supergiant Games"});
MERGE (:Studio {name: "Santa Monica Studio"});
MERGE (:Studio {name: "Team Cherry"});

// Piattaforme
MERGE (:Platform {name: "PC"});
MERGE (:Platform {name: "PlayStation 5"});
MERGE (:Platform {name: "PlayStation 4"});
MERGE (:Platform {name: "Xbox Series X"});
MERGE (:Platform {name: "Nintendo Switch"});

// Meccaniche
MERGE (:Mechanic {name: "Spirit Ashes"});
MERGE (:Mechanic {name: "Parry System"});
MERGE (:Mechanic {name: "Posture System"});
MERGE (:Mechanic {name: "Dialogue Choices"});
MERGE (:Mechanic {name: "Turn-Based Combat"});
MERGE (:Mechanic {name: "Permadeath"});
MERGE (:Mechanic {name: "Dodge Roll"});
MERGE (:Mechanic {name: "Stealth"});

// ==========================================
// 3. I GIOCHI
// ==========================================
MERGE (:Game {name: "Elden Ring", release_year: 2022});
MERGE (:Game {name: "Dark Souls III", release_year: 2016});
MERGE (:Game {name: "Sekiro", release_year: 2019});
MERGE (:Game {name: "Bloodborne", release_year: 2015});
MERGE (:Game {name: "Cyberpunk 2077", release_year: 2020});
MERGE (:Game {name: "Baldur's Gate 3", release_year: 2023});
MERGE (:Game {name: "Resident Evil 2 Remake", release_year: 2023});
MERGE (:Game {name: "Hades", release_year: 2020});
MERGE (:Game {name: "God of War", release_year: 2018});
MERGE (:Game {name: "Hollow Knight", release_year: 2017});

// ==========================================
// 4. I BOSS
// ==========================================
MERGE (:Boss {name: "Malenia", difficulty: 10});
MERGE (:Boss {name: "Starscourge Radahn", difficulty: 9});
MERGE (:Boss {name: "Nameless King", difficulty: 9});
MERGE (:Boss {name: "Orphan of Kos", difficulty: 10});
MERGE (:Boss {name: "Genichiro Ashina", difficulty: 8});
MERGE (:Boss {name: "Isshin Sword Saint", difficulty: 10});
MERGE (:Boss {name: "Mr. X", difficulty: 7});
MERGE (:Boss {name: "Baldur", difficulty: 7});
MERGE (:Boss {name: "The Radiance", difficulty: 9});
MERGE (:Boss {name: "Adam Smasher", difficulty: 6});
MERGE (:Boss {name: "Ketheric Thorm", difficulty: 7});
MERGE (:Boss {name: "Hades (Boss)", difficulty: 8});

// ==========================================
// 5. RELAZIONI: GENERI E STUDIOS
// ==========================================
MATCH (er:Game {name:"Elden Ring"}), (ds3:Game {name:"Dark Souls III"}), (bb:Game {name:"Bloodborne"})
MATCH (soul:Genre {name:"Soulslike"}), (from:Studio {name:"FromSoftware"})
MERGE (er)-[:PART_OF_GENRE]->(soul) MERGE (er)-[:DEVELOPED_BY]->(from)
MERGE (ds3)-[:PART_OF_GENRE]->(soul) MERGE (ds3)-[:DEVELOPED_BY]->(from)
MERGE (bb)-[:PART_OF_GENRE]->(soul) MERGE (bb)-[:DEVELOPED_BY]->(from);

MATCH (sek:Game {name:"Sekiro"}), (act:Genre {name:"Action"}), (from:Studio {name:"FromSoftware"})
MERGE (sek)-[:PART_OF_GENRE]->(act) MERGE (sek)-[:DEVELOPED_BY]->(from);

MATCH (cp:Game {name:"Cyberpunk 2077"}), (ow:Genre {name:"Open World"}), (cdpr:Studio {name:"CD Projekt Red"})
MERGE (cp)-[:PART_OF_GENRE]->(ow) MERGE (cp)-[:DEVELOPED_BY]->(cdpr);

MATCH (bg3:Game {name:"Baldur's Gate 3"}), (rpg:Genre {name:"Action RPG"}), (lar:Studio {name:"Larian Studios"})
MERGE (bg3)-[:PART_OF_GENRE]->(rpg) MERGE (bg3)-[:DEVELOPED_BY]->(lar);

MATCH (re2:Game {name:"Resident Evil 2 Remake"}), (surv:Genre {name:"Survival Horror"}), (cap:Studio {name:"Capcom"})
MERGE (re2)-[:PART_OF_GENRE]->(surv) MERGE (re2)-[:DEVELOPED_BY]->(cap);

MATCH (hades:Game {name:"Hades"}), (rogue:Genre {name:"Roguelike"}), (super:Studio {name:"Supergiant Games"})
MERGE (hades)-[:PART_OF_GENRE]->(rogue) MERGE (hades)-[:DEVELOPED_BY]->(super);

MATCH (gow:Game {name:"God of War"}), (rpg:Genre {name:"Action RPG"}), (santa:Studio {name:"Santa Monica Studio"})
MERGE (gow)-[:PART_OF_GENRE]->(rpg) MERGE (gow)-[:DEVELOPED_BY]->(santa);

MATCH (hk:Game {name:"Hollow Knight"}), (metr:Genre {name:"Metroidvania"}), (team:Studio {name:"Team Cherry"})
MERGE (hk)-[:PART_OF_GENRE]->(metr) MERGE (hk)-[:DEVELOPED_BY]->(team);

// ==========================================
// 6. RELAZIONI: BOSS E MECCANICHE
// ==========================================
MATCH (er:Game {name:"Elden Ring"}), (mal:Boss {name:"Malenia"}), (rad:Boss {name:"Starscourge Radahn"}), (ash:Mechanic {name:"Spirit Ashes"})
MERGE (er)-[:HAS_BOSS]->(mal) MERGE (er)-[:HAS_BOSS]->(rad) MERGE (er)-[:USES_MECHANIC]->(ash);

MATCH (ds3:Game {name:"Dark Souls III"}), (nam:Boss {name:"Nameless King"}), (roll:Mechanic {name:"Dodge Roll"})
MERGE (ds3)-[:HAS_BOSS]->(nam) MERGE (ds3)-[:USES_MECHANIC]->(roll);

MATCH (sek:Game {name:"Sekiro"}), (gen:Boss {name:"Genichiro Ashina"}), (iss:Boss {name:"Isshin Sword Saint"}), (post:Mechanic {name:"Posture System"})
MERGE (sek)-[:HAS_BOSS]->(gen) MERGE (sek)-[:HAS_BOSS]->(iss) MERGE (sek)-[:USES_MECHANIC]->(post);

MATCH (bb:Game {name:"Bloodborne"}), (orp:Boss {name:"Orphan of Kos"}), (par:Mechanic {name:"Parry System"})
MERGE (bb)-[:HAS_BOSS]->(orp) MERGE (bb)-[:USES_MECHANIC]->(par);

MATCH (cp:Game {name:"Cyberpunk 2077"}), (sma:Boss {name:"Adam Smasher"}), (dial:Mechanic {name:"Dialogue Choices"}), (ste:Mechanic {name:"Stealth"})
MERGE (cp)-[:HAS_BOSS]->(sma) MERGE (cp)-[:USES_MECHANIC]->(dial) MERGE (cp)-[:USES_MECHANIC]->(ste);

MATCH (bg3:Game {name:"Baldur's Gate 3"}), (keth:Boss {name:"Ketheric Thorm"}), (turn:Mechanic {name:"Turn-Based Combat"})
MERGE (bg3)-[:HAS_BOSS]->(keth) MERGE (bg3)-[:USES_MECHANIC]->(turn);

MATCH (re2:Game {name:"Resident Evil 2 Remake"}), (mrx:Boss {name:"Mr. X"}), (par:Mechanic {name:"Parry System"})
MERGE (re2)-[:HAS_BOSS]->(mrx) MERGE (re2)-[:USES_MECHANIC]->(par);

MATCH (hades:Game {name:"Hades"}), (had:Boss {name:"Hades (Boss)"}), (perm:Mechanic {name:"Permadeath"})
MERGE (hades)-[:HAS_BOSS]->(had) MERGE (hades)-[:USES_MECHANIC]->(perm);

MATCH (gow:Game {name:"God of War"}), (bal:Boss {name:"Baldur"}), (par:Mechanic {name:"Parry System"})
MERGE (gow)-[:HAS_BOSS]->(bal) MERGE (gow)-[:USES_MECHANIC]->(par);

MATCH (hk:Game {name:"Hollow Knight"}), (radia:Boss {name:"The Radiance"}), (roll:Mechanic {name:"Dodge Roll"})
MERGE (hk)-[:HAS_BOSS]->(radia) MERGE (hk)-[:USES_MECHANIC]->(roll);

// ==========================================
// 7. RELAZIONI: PIATTAFORME
// ==========================================
MATCH (pc:Platform {name:"PC"}), (ps5:Platform {name:"PlayStation 5"}), (ps4:Platform {name:"PlayStation 4"}), (xbox:Platform {name:"Xbox Series X"}), (switch:Platform {name:"Nintendo Switch"})

MATCH (er:Game {name:"Elden Ring"}) MERGE (er)-[:AVAILABLE_ON]->(pc) MERGE (er)-[:AVAILABLE_ON]->(ps5) MERGE (er)-[:AVAILABLE_ON]->(ps4) MERGE (er)-[:AVAILABLE_ON]->(xbox);
MATCH (ds3:Game {name:"Dark Souls III"}) MERGE (ds3)-[:AVAILABLE_ON]->(pc) MERGE (ds3)-[:AVAILABLE_ON]->(ps4) MERGE (ds3)-[:AVAILABLE_ON]->(xbox);
MATCH (sek:Game {name:"Sekiro"}) MERGE (sek)-[:AVAILABLE_ON]->(pc) MERGE (sek)-[:AVAILABLE_ON]->(ps4) MERGE (sek)-[:AVAILABLE_ON]->(xbox);
MATCH (bb:Game {name:"Bloodborne"}) MERGE (bb)-[:AVAILABLE_ON]->(ps4);
MATCH (cp:Game {name:"Cyberpunk 2077"}) MERGE (cp)-[:AVAILABLE_ON]->(pc) MERGE (cp)-[:AVAILABLE_ON]->(ps5) MERGE (cp)-[:AVAILABLE_ON]->(ps4) MERGE (cp)-[:AVAILABLE_ON]->(xbox);
MATCH (bg3:Game {name:"Baldur's Gate 3"}) MERGE (bg3)-[:AVAILABLE_ON]->(pc) MERGE (bg3)-[:AVAILABLE_ON]->(ps5) MERGE (bg3)-[:AVAILABLE_ON]->(xbox);
MATCH (re2:Game {name:"Resident Evil 2 Remake"}) MERGE (re2)-[:AVAILABLE_ON]->(pc) MERGE (re2)-[:AVAILABLE_ON]->(ps5) MERGE (re2)-[:AVAILABLE_ON]->(ps4) MERGE (re2)-[:AVAILABLE_ON]->(xbox);
MATCH (hades:Game {name:"Hades"}) MERGE (hades)-[:AVAILABLE_ON]->(pc) MERGE (hades)-[:AVAILABLE_ON]->(switch) MERGE (hades)-[:AVAILABLE_ON]->(ps5) MERGE (hades)-[:AVAILABLE_ON]->(ps4) MERGE (hades)-[:AVAILABLE_ON]->(xbox);
MATCH (gow:Game {name:"God of War"}) MERGE (gow)-[:AVAILABLE_ON]->(pc) MERGE (gow)-[:AVAILABLE_ON]->(ps4);
MATCH (hk:Game {name:"Hollow Knight"}) MERGE (hk)-[:AVAILABLE_ON]->(pc) MERGE (hk)-[:AVAILABLE_ON]->(switch) MERGE (hk)-[:AVAILABLE_ON]->(ps4) MERGE (hk)-[:AVAILABLE_ON]->(xbox);

// ==========================================
// 8. SIMILARITY TRA GIOCHI
// ==========================================
MATCH (a:Game {name:"Elden Ring"}), (b:Game {name:"Dark Souls III"}) MERGE (a)-[:SIMILAR_TO]->(b);
MATCH (a:Game {name:"Elden Ring"}), (b:Game {name:"Sekiro"}) MERGE (a)-[:SIMILAR_TO]->(b);
MATCH (a:Game {name:"Elden Ring"}), (b:Game {name:"Bloodborne"}) MERGE (a)-[:SIMILAR_TO]->(b);
MATCH (a:Game {name:"God of War"}), (b:Game {name:"Resident Evil 2 Remake"}) MERGE (a)-[:SIMILAR_TO]->(b);
MATCH (a:Game {name:"Hollow Knight"}), (b:Game {name:"Bloodborne"}) MERGE (a)-[:SIMILAR_TO]->(b);
MATCH (a:Game {name:"Hades"}), (b:Game {name:"Hollow Knight"}) MERGE (a)-[:SIMILAR_TO]->(b);

// ==========================================
// 9. BLOG POST DI SEED (solo review)
// ==========================================
MERGE (p1:BlogPost {title: "Why Elden Ring Changed Open World Design"})
ON CREATE SET p1.type = "review", p1.angle = "open world design", p1.created_at = datetime();

MATCH (g1:Game {name:"Elden Ring"}), (p1:BlogPost {title:"Why Elden Ring Changed Open World Design"})
MERGE (g1)-[:COVERED_IN]->(p1);

// Claims e fonti di seed
MERGE (c1:Claim {text: "Elden Ring ha ridefinito il design open-world nei soulslike."});
MATCH (p1:BlogPost {title:"Why Elden Ring Changed Open World Design"}), (c1:Claim {text:"Elden Ring ha ridefinito il design open-world nei soulslike."})
MERGE (p1)-[:CLAIMS]->(c1);

MERGE (s1:Source {url: "https://www.ign.com/articles/elden-ring-review"});
MATCH (p1:BlogPost {title:"Why Elden Ring Changed Open World Design"}), (s1:Source {url:"https://www.ign.com/articles/elden-ring-review"})
MERGE (p1)-[:USED_SOURCE]->(s1);

// ==========================================
// VERIFICA FINALE
// ==========================================
MATCH (n)
RETURN labels(n) AS Labels, count(*) AS Count
ORDER BY Count DESC;
