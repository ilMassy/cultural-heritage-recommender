"""
test_recommender_clip_scoring.py — Test unitari per la parte di recommender_clip.py
che non richiede GPU/modello (punto 11 della roadmap): minmax(), score_users(),
top_k_ids(). Non testa encode_text/il caricamento del modello CLIP — quella parte
richiede transformers/torch e va verificata solo manualmente (nessun modo economico
di testarla senza scaricare il modello).

Uso: pytest src/test_recommender_clip_scoring.py -v
"""

import numpy as np
import pytest

from recommender_clip import minmax, score_users, top_k_ids, period_boost, build_prompt


# ------------------------------------------------------------------------ minmax

def test_minmax_scales_each_row_to_0_1():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    out = minmax(x)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_minmax_is_per_row_not_global():
    # riga 0 ha range [0,10], riga 1 ha range [100,110]: ogni riga si deve
    # normalizzare sul PROPRIO range, non su quello globale della matrice
    x = np.array([
        [0.0, 5.0, 10.0],
        [100.0, 105.0, 110.0],
    ])
    out = minmax(x)
    assert out[0].max() == pytest.approx(1.0)
    assert out[0].min() == pytest.approx(0.0)
    assert out[1].max() == pytest.approx(1.0)
    assert out[1].min() == pytest.approx(0.0)
    # il valore centrale di ogni riga deve mappare a 0.5 (range simmetrico)
    assert out[0, 1] == pytest.approx(0.5)
    assert out[1, 1] == pytest.approx(0.5)


def test_minmax_constant_row_does_not_divide_by_zero():
    # una riga costante (hi - lo = 0): non deve dare NaN/inf grazie al maximum(...,1e-12)
    x = np.array([[5.0, 5.0, 5.0]])
    out = minmax(x)
    assert np.all(np.isfinite(out))


# ------------------------------------------------------------------- score_users

def test_score_users_alpha_zero_uses_only_text():
    # alpha=0 -> score deve dipendere solo da txt, non da img
    Q = np.array([[1.0, 0.0]])
    img = np.array([[1.0, 0.0], [0.0, 1.0]])  # item 0 identico a Q, item 1 ortogonale
    txt = np.array([[0.0, 1.0], [1.0, 0.0]])  # invertito rispetto a img
    boosts = np.zeros((1, 2))
    scores = score_users(Q, img, txt, boosts, alpha=0.0, period_weight=0.0)
    # con alpha=0 e period_weight=0: score = minmax(Q @ txt.T)
    # Q@txt.T = [0*1, 1*... ] -> item1 (txt=[1,0]) ha coseno 1 con Q, item0 (txt=[0,1]) ha coseno 0
    assert scores[0, 1] > scores[0, 0]  # item 1 deve vincere, non item 0


def test_score_users_alpha_one_uses_only_image():
    Q = np.array([[1.0, 0.0]])
    img = np.array([[1.0, 0.0], [0.0, 1.0]])
    txt = np.array([[0.0, 1.0], [1.0, 0.0]])
    boosts = np.zeros((1, 2))
    scores = score_users(Q, img, txt, boosts, alpha=1.0, period_weight=0.0)
    # con alpha=1: score = minmax(Q @ img.T) -> item0 (img=[1,0]) vince, non item1
    assert scores[0, 0] > scores[0, 1]


def test_score_users_period_weight_one_uses_only_boost():
    # period_weight=1 -> il contenuto CLIP (testo/immagine) non deve influenzare lo score
    Q = np.array([[1.0, 0.0]])
    img = np.array([[1.0, 0.0], [0.0, 1.0]])
    txt = np.array([[1.0, 0.0], [0.0, 1.0]])
    boosts = np.array([[0.2, 0.9]])  # item 1 ha boost periodo più alto
    scores = score_users(Q, img, txt, boosts, alpha=0.5, period_weight=1.0)
    np.testing.assert_allclose(scores, boosts)


def test_score_users_output_shape():
    n_users, n_items, dim = 3, 7, 512
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(n_users, dim))
    img = rng.normal(size=(n_items, dim))
    txt = rng.normal(size=(n_items, dim))
    boosts = rng.uniform(size=(n_users, n_items))
    scores = score_users(Q, img, txt, boosts, alpha=0.5, period_weight=0.15)
    assert scores.shape == (n_users, n_items)


def test_score_users_matches_manual_formula():
    # score = (1-w_p) * [alpha*s_img + (1-alpha)*s_txt] + w_p*boost, con s_img/s_txt
    # già minmax-ati per riga — verifica end-to-end contro un calcolo fatto a mano
    Q = np.array([[1.0, 0.0]])
    img = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    txt = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    boosts = np.array([[0.1, 0.9, 0.5]])
    alpha, w_p = 0.3, 0.2

    s_img_manual = minmax(Q @ img.T)
    s_txt_manual = minmax(Q @ txt.T)
    expected = (1 - w_p) * (alpha * s_img_manual + (1 - alpha) * s_txt_manual) + w_p * boosts

    scores = score_users(Q, img, txt, boosts, alpha=alpha, period_weight=w_p)
    np.testing.assert_allclose(scores, expected)


# ------------------------------------------------------------------- top_k_ids

def test_top_k_ids_returns_correct_order():
    item_ids = ["a", "b", "c", "d"]
    scores_row = np.array([0.1, 0.9, 0.5, 0.3])
    result = top_k_ids(scores_row, item_ids, k=2)
    assert result == ["b", "c"]  # ordinati per score decrescente


def test_top_k_ids_handles_k_larger_than_items():
    item_ids = ["a", "b"]
    scores_row = np.array([0.5, 0.9])
    result = top_k_ids(scores_row, item_ids, k=10)
    assert result == ["b", "a"]


def test_top_k_ids_stable_on_ties():
    # a parità di score, l'ordine deve restare quello originale (argsort kind="stable")
    item_ids = ["a", "b", "c"]
    scores_row = np.array([0.5, 0.5, 0.9])
    result = top_k_ids(scores_row, item_ids, k=3)
    assert result == ["c", "a", "b"]


# --------------------------------------------------------------- period_boost

def test_period_boost_zero_distance_is_one():
    assert period_boost(1800, 1800, 50) == pytest.approx(1.0)


def test_period_boost_unknown_year_is_neutral():
    assert period_boost(None, 1800, 50) == pytest.approx(0.5)


def test_period_boost_clamped_at_zero():
    # distanza (500) maggiore della bandwidth (50): non deve andare negativo
    assert period_boost(1300, 1800, 50) == 0.0


def test_period_boost_matches_ground_truth_formula():
    # stessa forma di score_item in generate_user_profiles.py (sez. 8 del report,
    # decisione metodologica citata esplicitamente)
    year, center, bandwidth = 1820, 1800, 50
    expected = max(0.0, 1.0 - abs(year - center) / bandwidth)
    assert period_boost(year, center, bandwidth) == pytest.approx(expected)


# ------------------------------------------------------------------- build_prompt

def test_build_prompt_includes_all_preference_types():
    profile = {
        "preferred_classifications": ["Paintings"],
        "preferred_nationalities": ["French", "Italian"],
        "preferred_tags": ["Men", "Portraits"],
    }
    prompt = build_prompt(profile)
    assert "paintings" in prompt.lower()
    assert "french" in prompt.lower() and "italian" in prompt.lower()
    assert "men" in prompt.lower() and "portraits" in prompt.lower()


def test_build_prompt_handles_missing_tags():
    # nessun tag preferito: non deve crashare né lasciare una virgola pendente
    profile = {
        "preferred_classifications": ["Paintings"],
        "preferred_nationalities": ["French"],
        "preferred_tags": [],
    }
    prompt = build_prompt(profile)
    assert not prompt.strip().endswith(",")


def test_build_prompt_handles_empty_profile():
    # profilo degenere (nessuna preferenza): non deve sollevare eccezioni
    profile = {"preferred_classifications": [], "preferred_nationalities": [], "preferred_tags": []}
    prompt = build_prompt(profile)
    assert isinstance(prompt, str)
