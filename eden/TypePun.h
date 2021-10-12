
// smuggle int32 into f32 position
// dodgy, but that's life
union TypePun_I32F32{
	int32_t i32; float f32;
	static_assert( sizeof(i32) == sizeof(f32), "Single-precision float must have same same size as int32_t for type punning to work" );
};
auto EncodeI32ToF32( int32_t i ){
	TypePun_I32F32 cast;
	cast.i32 = i;
	return cast.f32;
}
auto EncodeF32ToI32( float f ){
	TypePun_I32F32 cast;
	cast.f32 = f;
	return cast.i32;
}
