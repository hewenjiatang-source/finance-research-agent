#!/bin/bash
# Continuous test loop - separate log per query, real-time monitoring

cd "$(dirname "$0")/.."
source .venv/bin/activate

QUERIES=(
    "2024年至2025年大模型Agent技术方向与后端系统开发方向的对比研究：检索两个领域的最新学术论文发表数量与趋势变化、工业界代表性落地案例与市场规模数据"
    "计算Transformer模型在7B、13B、70B三种参数量下的训练FLOPs和推理显存占用，并检索2024年至2025年主流大模型的实际部署成本和推理延迟数据"
    "中国新能源汽车行业2024年至2025年的市场份额变化、主要品牌销量排名、电池技术路线（磷酸铁锂vs三元锂）的技术对比与成本分析"
    "检索近一年（2024年至2025年）关于RLHF的顶级会议论文，统计NeurIPS、ICML、ICLR各会议收录数量，并分析该领域的技术演进趋势"
    "对比分析2024年至2025年生成式AI在医疗健康领域和金融投资领域的应用进展：检索各领域的代表性产品、监管政策变化、以及商业化落地案例"
)

IDX=1
for Q in "${QUERIES[@]}"; do
    echo ""
    echo "========================================"
    echo "[Test $IDX] Starting: ${Q:0:60}..."
    echo "========================================"

    # Clean up environment
    # rm -f data/memory.db
    # rm -rf "outputs/reports/test${IDX}"
    mkdir -p "outputs/reports/test${IDX}"

    LOG="outputs/test_${IDX}.log"

    # Run (foreground, output visible in real time)
    PYTHONUNBUFFERED=1 python scripts/run_research.py \
        --query "$Q" \
        --output_dir "outputs/reports/test${IDX}" \
        > "$LOG" 2>&1

    # Extract key metrics
    echo ""
    echo "[Test $IDX] Result summary:"
    grep -E "DAG 生成完成|子任务完成|报告生成完成|耗时:|置信度" "$LOG" | tail -8 || true

    # Check report file
    REPORT=$(find "outputs/reports/test${IDX}" -name "*.md" | head -1)
    if [ -f "$REPORT" ]; then
        LEN=$(wc -c < "$REPORT")
        echo "[Test $IDX] Report: $REPORT ($LEN bytes)"
        # Check whether the report is empty/failed
        if grep -q "Research failed" "$REPORT"; then
            echo "[Test $IDX] ⚠️ Report marked as failed"
        else
            echo "[Test $IDX] ✅ Report generated successfully"
        fi
    else
        echo "[Test $IDX] ❌ Report file not generated"
    fi

    # Short pause to avoid API rate limits
    sleep 10

    IDX=$((IDX + 1))
done

echo ""
echo "========================================"
echo "All tests completed"
echo "========================================"
