import argparse

import nifty
import vigra
from elf.io import open_file
from elf.segmentation import get_multicut_solver


def solve_last_multicut(solver_name, time_limit):
    n_threads = 16
    solver = get_multicut_solver(solver_name)

    path = "/scratch/pape/jils_project/full_seg/tmp/data.n5"
    out_path = "/scratch/pape/jils_project/full_seg/data.n5"
    out_key = f"node_labels/alternative/{solver_name}"

    print("Load edges and costs from", path)
    with open_file(path, "r") as f:
        graph_group = f["s1/graph"]
        ignore_label = graph_group.attrs["ignore_label"]

        ds = graph_group["edges"]
        ds.n_threads = n_threads
        uv_ids = ds[:]
        n_edges = len(uv_ids)
        n_nodes = int(uv_ids.max() + 1)

        ds = f["s1/costs"]
        ds.n_threads = n_threads
        costs = ds[:]
        assert len(costs) == n_edges, "%i, %i" % (len(costs), n_edges)

        ds = f["s1/node_labeling"]
        ds.n_threads = n_threads
        initial_node_labeling = ds[:]

    print("Build the graph")
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)

    print("Start multicut solver")
    node_labeling = solver(graph, costs, n_threads=n_threads, time_limit=time_limit)

    print("Relabel nodes")
    initial_node_labeling = node_labeling[initial_node_labeling]
    # make sure zero is mapped to 0 if we have an ignore label
    n_nodes = len(node_labeling)
    if ignore_label and node_labeling[0] != 0:
        new_max_label = int(initial_node_labeling.max() + 1)
        initial_node_labeling[initial_node_labeling == 0] = new_max_label
        initial_node_labeling[0] = 0
    vigra.analysis.relabelConsecutive(initial_node_labeling, start_label=1, keep_zeros=True, out=initial_node_labeling)

    print("Save node labels to", out_path, out_key)
    # write node labeling
    node_shape = (n_nodes,)
    chunks = (min(n_nodes, 524288),)
    with open_file(out_path) as f:
        ds = f.require_dataset(out_key, dtype="uint64", shape=node_shape, chunks=chunks, compression="gzip")
        ds.n_threads = n_threads
        ds[:] = initial_node_labeling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver_name", "-s")
    parser.add_argument("--time_limit", "-t", default=None)
    args = parser.parse_args()
    solve_last_multicut(args.solver_name, args.time_limit)


if __name__ == "__main__":
    main()
