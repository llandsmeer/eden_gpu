// helpers

template< typename Container >
static void AppendToVector(Container &append_to, const Container &append_this){
	append_to.insert(append_to.end(), append_this.begin(), append_this.end());
};
template<
	typename CAppendTo, typename CAppendThis,
	typename std::enable_if< !std::is_same< CAppendTo, CAppendThis >::value, int >::type = 0
>
static void AppendToVector(CAppendTo &append_to, const CAppendThis &append_this){
	auto new_size = append_to.size() + append_this.size() ;
	append_to.reserve( new_size );
	for( size_t i = 0; i < append_this.size(); i++ ){
		append_to.push_back( append_this[i] );
	}
};
