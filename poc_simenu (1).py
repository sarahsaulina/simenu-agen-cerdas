"""PoC SIMENU: solver ILP + verifier (data sampel ilustrasi, bukan data pasar/TKPI).
Jalankan: pip install numpy scipy pandas ; python poc_simenu.py"""
import math, numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

recipes = [
 # id, nama, protein_utama, cost, menit, kcal/porsi, protein g/porsi, pref, alergen, bahan
 ("R01","Ayam kecap + tumis buncis","ayam",38500,35,520,32,5,[],"ayam kecap manis bawang putih buncis"),
 ("R02","Tempe orek + sayur bayam","tempe",21000,25,480,22,5,[],"tempe kecap cabai bayam bawang merah"),
 ("R03","Soto ayam bening","ayam",36000,40,410,34,5,[],"ayam kunyit serai kentang wortel kol bihun"),
 ("R04","Ikan nila bakar + lalapan","ikan",42000,35,500,38,5,["ikan"],"ikan nila kecap jeruk nipis timun selada"),
 ("R05","Telur balado + capcay","telur",28500,30,530,24,5,["telur"],"telur cabai tomat sayur capcay wortel kol"),
 ("R06","Nasi goreng sayur + telur","telur",24000,20,560,20,5,["telur"],"nasi telur bawang putih wortel buncis kecap"),
 ("R07","Sayur lodeh + tahu","tahu",33000,40,470,22,5,[],"tahu santan labu siam terong kacang panjang"),
 ("R08","Pepes tahu","tahu",25000,30,430,21,4,[],"tahu daun kemangi tomat bawang daun pisang"),
 ("R09","Tahu bacem + tumis kangkung","tahu",22000,40,450,20,4,[],"tahu kecap gula merah kangkung bawang putih"),
 ("R10","Nasi goreng sayur-tempe","tempe",20000,20,540,20,4,[],"nasi tempe bawang putih wortel buncis kecap"),
 ("R11","Tumis udang buncis","udang",30000,25,400,26,5,["udang"],"udang buncis bawang putih saus tiram"),
 ("R12","Sambal goreng udang","udang",45000,30,450,30,4,["udang"],"udang santan cabai tomat bawang"),
 ("R13","Rendang daging sapi","sapi",68000,120,620,36,5,[],"daging sapi santan cabai serai lengkuas"),
 ("R14","Gado-gado","tahu",26000,25,480,18,3,["kacang"],"tahu tempe kangkung tauge kentang bumbu kacang"),
 ("R15","Pecel lele","ikan",35000,35,520,30,3,["ikan","kacang"],"lele goreng sambal pecel kacang kol timun"),
 ("R16","Ayam goreng lengkuas","ayam",41000,45,560,33,4,[],"ayam lengkuas bawang putih ketumbar"),
 ("R17","Oseng tempe kemangi","tempe",19000,20,430,19,3,[],"tempe kemangi cabai bawang merah tomat"),
]

df = pd.DataFrame(recipes, columns=["id","nama","protein_utama","cost","menit","kcal","protein","pref","alergen","bahan"])

def solve(budget, alergen=(), hari=7, tmax=40, kmin=400, kmax=700, pmin=20, mode="pref"):
    n = len(df); cost = df["cost"].to_numpy(float); pref = df["pref"].to_numpy(float)
    c = -pref*100 + cost/1000.0 if mode == "pref" else cost      # maks preferensi | min biaya
    ub = np.array([0 if (set(a) & set(alergen)) or m > tmax else 1 for a, m in zip(df["alergen"], df["menit"])], float)
    A = [np.ones(n), cost, df["kcal"].to_numpy(float)-kmin, df["kcal"].to_numpy(float)-kmax, df["protein"].to_numpy(float)-pmin]
    lo = [hari, -np.inf, 0, -np.inf, 0]
    hi = [hari, budget if mode == "pref" else np.inf, np.inf, 0, np.inf]
    for g in df["protein_utama"].unique():                         # variasi protein utama
        A.append((df["protein_utama"] == g).to_numpy(float)); lo.append(-np.inf); hi.append(math.ceil(hari/3))
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(n), bounds=Bounds(np.zeros(n), ub))
    return df[np.round(res.x) == 1] if res.success else None

def verifikasi(plan, budget, alergen=(), hari=7, tmax=40, kmin=400, kmax=700, pmin=20):
    if plan is None: return False
    return bool(len(plan) == hari and plan["cost"].sum() <= budget
                and not any(set(a) & set(alergen) for a in plan["alergen"])
                and (plan["menit"] <= tmax).all() and kmin <= plan["kcal"].mean() <= kmax
                and plan["protein"].mean() >= pmin
                and (plan["protein_utama"].value_counts() <= math.ceil(hari/3)).all())

def rp(x): return "Rp" + f"{x:,.0f}".replace(",", ".")

if __name__ == "__main__":
    for nama, b, al in [("A", 250000, ("udang",)), ("B", 200000, ("udang","telur")), ("C", 120000, ("udang",))]:
        p = solve(b, al)
        if p is None:
            m = solve(10**9, al, mode="biaya")
            print(f"Skenario {nama}: INFEASIBLE pada {rp(b)}; anggaran minimum {rp(m['cost'].sum())}")
        else:
            print(f"Skenario {nama}: total {rp(p['cost'].sum())} | verifier lolos: {verifikasi(p, b, al)}")
            print("  " + ", ".join(p["nama"]))
