datasets="reddit_thread github_stargazers twitch_ego_net deezer_ego_net"
models="GraphConv SAGEConv GCNConv ResGatedGraphConv TransformerConv GATConv"
experiments="ac"

echo "Dataset List: $datasets"

for dataset in $datasets; do
    echo "Processing Dataset: $dataset"
    for model in $models; do
        echo "  Current Model: $model"
        for exp in $experiments; do
            echo "    Experiment Type: $exp"
            python main_extension.py --dataset "$dataset" --model "$model" --exp "$exp"
        done
    done
done

echo "All tasks are done!"