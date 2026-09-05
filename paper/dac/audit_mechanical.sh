#!/usr/bin/env bash
set -u

run_check() {
  local name="$1"
  shift
  echo "===== ${name} ====="
  "$@" || true
}

run_check M1_ASCII grep -rn -- '---' sections/*.tex
echo '===== M1_UNICODE ====='
grep -rn $'—\|–' sections/*.tex || true
run_check M11 grep -rnoE '\b(is|are|was|were|be|been|being)\s+([a-z]+ed|done|made|shown|given|taken|held|built|drawn|chosen|written|known|found|seen|set|put|sent|kept|met|run|used|based)\b' sections/*.tex
run_check M2 grep -rnoE ',? not [0-9A-Za-z\\]|not only .* but|rather than|less .* than|is the point|whatever it is|means nothing|more than just|not in competition with|on one hand|on the other hand' sections/*.tex
run_check M3_M4_M8_M9 grep -rnoE 'in effect|in a sense|at (its|the) (heart|core)|in essence|\btruly\b|\bgenuinely\b|\bindeed\b|\bin fact\b|precisely because|a testament to|the kind of .* that|exactly the kind|is the point|set(s)? .* apart|no (predecessor|one) .* (made|posed)|the key (insight|idea) is|the machine that|draw(s)? .* power from|under the hood|where .* meets' sections/*.tex
run_check M6 grep -rnE '^(Moreover|Furthermore|Additionally|Notably|Importantly|Indeed|Ultimately|Crucially|In turn|That said)' sections/*.tex
run_check M5_M16 grep -rnoiE '\bnovel\b|\bsignificant\b|\bsubstantial\b|\bimpressive\b|\bpromising\b|\bcomprehensive\b|\brobust\b|\bpowerful\b|\bseamless|\bcrucial\b|\bparadigm\b|\bleverag|\butiliz|\bfinaliz|[a-z]+-oriented\b|\bfactor\b|\bfeature[ds]?\b|\bmeaningful\b|\binsightful\b|\bprestigious\b|\bpossess|\bcontact(s|ed|ing)?\b|\bcurrently\b|\bimpact(s|ed|ing)?\b' sections/*.tex
run_check M10 grep -rnoiE 'promises to|stands? to|is poised to|opens the door to|is set to|has the potential to|keeps .* from|stands? in the way|\bunlocks?\b' sections/*.tex
run_check M17 grep -rnoiE '\b(pit(s|ted|ting)?|dispatch(es|ed|ing)?|chip(s|ped|ping)? (away )?at|marshal(s|led|ling)?|orchestrat(e|es|ed|ing)|wrangl(e|es|ed|ing)|harness(es|ed|ing)?|forge[sd]?|weav(e|es|ed|ing)|delv(e|es|ed|ing) into|usher(s|ed)? in|grappl(e|es|ed|ing) with|anew|afresh)\b' sections/*.tex
run_check M18 grep -rnE 'In this (paper|section), we' sections/*.tex
run_check M12 grep -rnoE 'the fact that|the question (as to |of )?whether|as to whether|in order to|there is no doubt but|the reason .* is because|owing to the fact that|in a [a-z]+ manner|is a (subject|man|woman) (that|who)|in the last analysis|along these lines|in terms of|one of the most' sections/*.tex
run_check M13 grep -rnoiE '\b(rather|very|pretty|little|quite|somewhat|fairly|certainly)\b' sections/*.tex
run_check M14 grep -rnoiE '\b(thusly|muchly|overly|firstly|secondly|thirdly)\b|[a-z]+wise\b' sections/*.tex
run_check M15 grep -rn '!' sections/*.tex
run_check PRECISION grep -rnoiE '\bcomprised of\b|\bdata is\b|different than|\bvery unique\b|\bdue to\b|\bless (than )?[0-9]' sections/*.tex
run_check DECOMP grep -rnoE '\b(three|four|five|six|seven)\b (stages|concerns|axes|requirements|dimensions|systems)' sections/*.tex

