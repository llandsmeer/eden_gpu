void print_eden_cli_header() {
	//print logo ^_^
    printf("       ###########    ###############              ###########    ######     #########       \n");
    printf("     ###############   ##################        ###############    #####  ##############    \n");
    printf("   #####          ####      #####    ######    #####          ####     ######       ######   \n");
    printf("  ####                    ####          ####  ####                     ####           #####  \n");
    printf("  ###############         ###            ###  ###############          ####            ####  \n");
    printf("  ############           ####            ###  ############             ###             ####  \n");
    printf("  ####                   ####            ###  ####                     ###             ####  \n");
    printf("  ####              #    ####           ####  ####              #     ####             ####  \n");
    printf("   #####          ###     ####         ####    #####          ###     ####            ####   \n");
    printf("    ################        ##############      ################     ####            ####    \n");
    printf("      ##########              ##########          ##########        ####            #####    \n\n");
    printf("--- Extensible Dynamics Engine for Networks ---\n");
	#ifndef BUILD_STAMP
	#define BUILD_STAMP __DATE__
	#endif
	printf("Build version " BUILD_STAMP "\n");
}
