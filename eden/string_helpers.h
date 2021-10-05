// LATER add a helper for single variables

// do not use default accuracy when converting numerics to alpha !
// expicitly specify what the alpha is used for
template< typename T, typename = typename std::enable_if< std::is_integral<T>::value >::type >
std::string accurate_string( T val ){
	return std::to_string( val );
}
std::string accurate_string( float val ){
	char tmps[100];
	sprintf(tmps, "%.9g", val);
	return tmps;
}
std::string accurate_string( double val ){
	char tmps[100];
	sprintf(tmps, "%.17g", val);
	return tmps;
}

std::string presentable_string( double val ){
	char tmps[100];
	sprintf(tmps, "%g", val);
	return tmps;
}
template< typename T, typename = typename std::enable_if< std::is_integral<T>::value >::type >
std::string presentable_string( T val ){
	char tmps[100];
	if( val < -1000000 || 1000000 < val ){
		// extreme values are probably packed refs
		sprintf(tmps, "0x%llx", (long long) val);
	}
	else{
		sprintf(tmps, "%lld", (long long) val);
	}
	
	return tmps;
}

template< typename T, typename std::enable_if< std::is_integral<T>::value, int >::type = 0 >
std::string itos(T val){
	return accurate_string(val);
}
template< typename T, typename std::enable_if< std::is_enum<T>::value, int >::type = 0 >
std::string itos(T val){
	return std::to_string(val);
}
