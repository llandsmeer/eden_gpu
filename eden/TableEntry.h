#ifndef EDEN_TABLEENTRY_H
#define EDEN_TABLEENTRY_H

// references to the raw tables
struct TabEntryRef{
	long long table;
	int entry;
	// could also use the compressed, encoded combination
};
typedef long long TabEntryRef_Packed;
static auto GetEncodedTableEntryId = []( long long global_idx_T_dest_table, long long entry_idx_T_dest ){
	// pack 1 trillion tables -> 16 million entries into 64bit indexes, upgrade if needed LATER
	
	const unsigned long long table_id = global_idx_T_dest_table * (1 << 24);
	const unsigned long long entry_id = entry_idx_T_dest % (1 << 24);
	const unsigned long long packed_id = table_id | entry_id ;
	
	return packed_id;
};
static auto GetDecodedTableEntryId = [  ]( unsigned long long packed_id ){
	long long global_idx_T_dest_table = packed_id >> 24;
	int entry_idx_T_dest = packed_id % (1 << 24);
	TabEntryRef ret = { global_idx_T_dest_table, entry_idx_T_dest };
	return ret;
};

#endif
