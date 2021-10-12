extern "C" {
// for strictly fixed-width data output (also useful for parallel writing to files)
struct FixedWidthNumberPrinter{
	int column_size;
	int delimiter_size;
	char delimiter_char;
	char format[50];
	
	int getNumberSize() const {
		return column_size - delimiter_size;// for separator
	}
	
	FixedWidthNumberPrinter( int _csize, char _delcha = ' ', int _dellen = 1 ){
		column_size = _csize;
		delimiter_size = _dellen;
		delimiter_char = _delcha;
		assert( column_size > delimiter_size );
		
		const int number_size = getNumberSize();
		const int digits = column_size - 3 - 5; // "+1.", "e+308"
		sprintf(format, "%%+%d.%dg", number_size, digits );
	}
	
	// should have length of column_size + terminator
	void write( float val, char *buf) const {
		const int number_size = getNumberSize();
		snprintf( buf, number_size + 1, format, val );
		// Also add some spaces, to delimit columns
		for( int i = 0; i < delimiter_size; i++ ){
			buf[number_size + i] = delimiter_char;
		}
		buf[number_size+1] = '\0';
	}
};
}
