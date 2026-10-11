% CARE-LG clingo program: hard / soft / filter rules, inheritance via parent/2, derivations via #show.
% PIPELINE P2 (+ P7 certificates). All numbers are integers scaled x1000 (same rounding as the NumPy mirror).
% Adapted from experiments_caren/knowledge/rules.lp + rules_v2.lp (resolution: deepest node wins, envelope
% on ties) with CARE-LG's hard predicate (src/graph/clinical_constraints.py) as the HARD tier.
%
% Facts supplied per query by src/knowledge_layer.py:
%   has_concept(C). parent(Child, Parent). depth(N, D).
%   frule(R, F, N). lo(R, L). hi(R, H).          feature rules (F = dataset feature, via its concept)
%   kind(F, K).  K in {equal, non_decreasing, reduce, increase, none}
%   eps(F, E). maxchg(F, M). horizon(H). use_horizon.
%   contra(R, Cls, N).                           contraindication rule R: class Cls at node N
%   src(F, X). cand(V). val(V, F, X).
%   inherit.                                     absent in the no-inheritance ablation

% ---- inheritance down the is-a hierarchy ----
active_node("ROOT").
active_node(C) :- has_concept(C).
active_node(A) :- inherit, active_node(C), parent(C, A).
anc(C, C) :- has_concept(C).
anc(A, C) :- anc(B, C), parent(B, A).

% ---- patient rule set: deepest applicable node wins; ties -> envelope ----
applicable(R, F) :- frule(R, F, N), active_node(N).
maxd(F, M) :- frule(_, F, _), M = #max{ D, R : applicable(R, F), frule(R, F, N), depth(N, D) }.
selected(R, F) :- applicable(R, F), frule(R, F, N), depth(N, D), maxd(F, D).
eff_lo(F, L) :- selected(_, F), L = #min{ X, R : selected(R, F), lo(R, X) }, L != #sup.
eff_hi(F, H) :- selected(_, F), H = #max{ X, R : selected(R, F), hi(R, X) }, H != #inf.
via(R, N, C) :- selected(R, F), frule(R, F, N), N != "ROOT", has_concept(C), inherit, anc(N, C).
via(R, N, C) :- selected(R, F), frule(R, F, N), N != "ROOT", has_concept(C), not inherit, N = C.
via(R, "ROOT", "ROOT") :- selected(R, F), frule(R, F, "ROOT").

% ---- HARD tier (CARE-LG predicate; never violated by a returned recourse) ----
hard(immutable, F, V)    :- cand(V), kind(F, equal), val(V, F, X), src(F, Y), X != Y.
hard(age_decrease, F, V) :- cand(V), kind(F, non_decreasing), val(V, F, X), src(F, Y), X < Y - 100.
hard(age_horizon, F, V)  :- cand(V), kind(F, non_decreasing), use_horizon, horizon(H), val(V, F, X), src(F, Y), X - Y > H.
hard(direction, F, V)    :- cand(V), kind(F, reduce), eps(F, E), val(V, F, X), src(F, Y), X - Y > E.
hard(direction, F, V)    :- cand(V), kind(F, increase), eps(F, E), val(V, F, X), src(F, Y), Y - X > E.

% ---- SOFT tier (penalised with a degree, never a veto) ----
dv(V, F, D) :- cand(V), val(V, F, X), eff_lo(F, L), D = L - X, D > 0.
dv(V, F, D) :- cand(V), val(V, F, X), eff_hi(F, H), D = X - H, D > 0.
soft(target, F, V, D)    :- dv(V, F, D).
soft(magnitude, F, V, D) :- cand(V), val(V, F, X), src(F, Y), maxchg(F, M), D = |X - Y| - M, D > 0.

% ---- FILTER tier: contraindicated intervention classes with derivation ----
contraindicated(R, Cls, N) :- contra(R, Cls, N), active_node(N).
blocked_via(R, Cls, N, C)  :- contraindicated(R, Cls, N), has_concept(C), inherit, anc(N, C).
blocked_via(R, Cls, N, C)  :- contraindicated(R, Cls, N), has_concept(C), not inherit, N = C.

% ---- passed checks for single-path certificates ----
check(immutable, F) :- kind(F, equal).
check(age_decrease, F) :- kind(F, non_decreasing).
check(direction, F) :- kind(F, reduce).
check(direction, F) :- kind(F, increase).
passed(K, F, V) :- cand(V), check(K, F), not hard(K, F, V).

#show selected/2.
#show eff_lo/2.
#show eff_hi/2.
#show via/3.
#show hard/3.
#show soft/4.
#show contraindicated/3.
#show blocked_via/4.
#show passed/3.
