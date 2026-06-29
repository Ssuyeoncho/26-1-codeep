#!/usr/bin/env bash
# 무인 배치: Phase 2(MNIST) + Phase 3(CIFAR)의 큐레이션 클래스 쌍을 한 번에 전부 실행.
# 자기 전에 한 줄로 돌려두는 용도.
#
#   bash run_all_pairs.sh
#   nohup bash run_all_pairs.sh > run_all_pairs.log 2>&1 &   # 로그아웃해도 계속 돌게
#
# 동작:
#   1) Phase 2 (MNIST, 가벼움)  — 쌍: 0v1, 1v8, 3v8, 4v9  (full, 단일 seed)
#   2) Phase 3 (CIFAR, 무거움) — 쌍: cat_vs_dog, airplane_vs_frog, deer_vs_horse, airplane_vs_automobile (fast)
#
# 한 쌍이 실패해도 다음 쌍/다음 phase로 계속 진행한다(각 쌍은 자체 결과 폴더에 저장).
# Phase 2를 먼저 끝내 가벼운 결과를 빨리 확보하고, 무거운 Phase 3를 뒤에 돌린다.
cd "$(dirname "$0")"

echo "########################################################################"
echo "# run_all_pairs : 시작 $(date '+%Y-%m-%d %H:%M:%S')"
echo "########################################################################"

echo ""
echo ">>> [1/2] Phase 2 (MNIST) pairs"
bash run_phase2.sh pairs || echo ">>> Phase 2 배치에서 일부/전체 문제 발생 — Phase 3로 계속 진행"

echo ""
echo ">>> [2/2] Phase 3 (CIFAR) pairs"
bash run_phase3.sh pairs || echo ">>> Phase 3 배치에서 일부/전체 문제 발생"

echo ""
echo "########################################################################"
echo "# run_all_pairs : 종료 $(date '+%Y-%m-%d %H:%M:%S')"
echo "#  결과: results/phase2/ , results/phase3/ (폴더명에 쌍·상태 표시)"
echo "########################################################################"
