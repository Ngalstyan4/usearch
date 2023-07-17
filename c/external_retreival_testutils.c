
#include "external_retreival_testutils.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int NUM_VECTORS = -1;
char** nodes;
int* node_sizes;
int* node_levels;

void print_level_histo(int* all_levels, int len) {
    for (int i = 0; i < len; i++) {
        fprintf(stderr, "level%d %d\n", i, all_levels[i]);
    }
}

char* extract_node(char* data, int progress, int dim, usearch_metadata_t* metadata, /*->>output*/ int* node_size,
                   int* level) {
    const int NODE_HEAD_BYTES = sizeof(usearch_label_t) + 4 /*sizeof dim */ + 4 /*sizeof level*/;
    const int VECTOR_BYTES = dim * sizeof(float);
    int node_bytes = 0;
    char* tape = data + progress;
    int read_dim_bytes = -1;
    memcpy(&read_dim_bytes, tape + sizeof(usearch_label_t), 4); //+sizeof(label)
    memcpy(level, tape + sizeof(usearch_label_t) + 4, 4);       //+sizeof(label)+sizeof(dim)
    assert(VECTOR_BYTES == read_dim_bytes);
    node_bytes += NODE_HEAD_BYTES + metadata->neighbors_base_bytes;
    node_bytes += metadata->neighbors_bytes * *level;
    node_bytes += VECTOR_BYTES;
    *node_size = node_bytes;
    return tape;
}

void prepare_external_index(char* mapped_index, int dim, size_t num_vectors, usearch_metadata_t* metadata) {
    NUM_VECTORS = num_vectors;

    int progress = 64; // sizeof file_header_t = 64
    // todo:: this is not quite dim in bytes. it is dim * sizeof(scalar_t) but for some reason punned_dense makes
    // scalar_t = byte_t since I do not quite understand the logic behind storing this on node_t, come back to look at
    // it when/if things break
    int all_levels[20] = {0};
    int level = 0;
    int node_size = 0;
    nodes = (char**)calloc(num_vectors, sizeof(char*));
    // node_sizes = (int*)calloc(num_vectors, sizeof(int));
    // node_levels = (int*)calloc(num_vectors, sizeof(int));
    assert(nodes != NULL);
    for (size_t i = 0; i < num_vectors; i++) {
        nodes[i] = extract_node(mapped_index, progress, dim, metadata, /*->>output*/ &node_size, &level);
        if ((size_t)level > sizeof(all_levels) / sizeof(int)) {
            fprintf(stderr, "warn: large level %d\n", level);
        } else {
            all_levels[level]++;
        }

        // node_sizes[i] = node_bytes;
        progress += node_size;
    }
    print_level_histo(all_levels, 20);
}

void free_external_index() { free(nodes); }

void* node_retriever(int id) {
    fprintf(stderr, "retrieving node %d\n", id);
    return nodes[id];
}
void* node_retriever_mut(int id) {
    fprintf(stderr, "mutating node %d\n", id);
    return nodes[id];
}
