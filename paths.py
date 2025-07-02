


def define_files(name,project_root):
    match name:
        case "parmareggio" | "parmigiano":
            scorre_path= str(project_root / "dataset_medi" / "Scorre_Parmareggio_no" / "*.png")
            base_shape_path = str(project_root / 'Schematics' / 'shapes' / 'parmareggio.png')
            base_print_path = str(project_root / 'Schematics' / 'prints' / 'parmareggio.png')
            recomposed_path = str(project_root / "Reconstructed" / "parmareggio_no.png")
        case "parmareggio_ok":
            scorre_path = str(project_root / "dataset_medi" / "Scorre_Parmareggio_ok" / "*.png")
            base_shape_path = str(project_root / 'Schematics' / 'shapes' / 'parmareggio.png')
            base_print_path = str(project_root / 'Schematics' / 'prints' / 'parmareggio.png')
            recomposed_path = str(project_root / "Reconstructed" / "parmareggio_ok.png")
        case "nappies":
            scorre_path = str(project_root / "dataset_medi" / "Scorre_nappies" / "*.png")
            base_shape_path = str(project_root / 'Schematics' / 'shapes' / 'nappies.png')
            base_print_path = str(project_root / 'Schematics' / 'prints' / 'nappies.png')
            recomposed_path = str(project_root / "Reconstructed" / "nappies.png")
        case "green_scratched":
            scorre_path = " "
            base_shape_path = str(project_root / 'Schematics' / 'shapes' /"green.png")
            base_print_path = str(project_root / 'Schematics' / 'prints' /"green.png")
            recomposed_path = str(project_root / "dataset_piccoli" / "dezoommata_green_cut.png")
        case "green_buco_in_piu":
            scorre_path = " str(project_root / 'dataset_piccoli' / 'Scorre_verde' / 'Buco_in_piu' / '*.png')"
            base_shape_path = str(project_root / 'Schematics' / 'shapes' / 'green.png')
            base_print_path = str(project_root / 'Schematics' / 'prints' / 'green.png')
            recomposed_path = str(project_root / "Reconstructed" / "green_buco_in_piu.png")

        case "green_buco_in_meno":
            scorre_path = str(project_root / 'dataset_piccoli' / 'Scorre_verde' / 'Buco_in_meno' / '*.png')
            base_shape_path = str(project_root / 'Schematics' / 'shapes' / 'green.png')
            base_print_path = str(project_root / 'Schematics' / 'prints' / 'green.png')
            recomposed_path = str(project_root / "Reconstructed" / "green_buco_in_meno.png")

        case "green_lettere_disallineate":
            scorre_path = str(project_root / 'dataset_piccoli' / 'Scorre_verde' / 'Lettere_disallineate' / '*.png')
            base_shape_path = str(project_root / 'Schematics' / 'shapes' / 'green.png')
            base_print_path = str(project_root / 'Schematics' / 'prints' / 'green.png')
            recomposed_path = str(project_root / "Reconstructed" / "green_lettere_disallineate.png")

        case "green_ok":
            scorre_path = str(project_root / 'dataset_piccoli' / 'Scorre_verde' / 'OK' / '*.png')
            base_shape_path = str(project_root / 'Schematics' / 'shapes' / 'green.png')
            base_print_path = str(project_root / 'Schematics' / 'prints' / 'green.png')
            recomposed_path = str(project_root / "Reconstructed" / "green_OK.png")
        case "nappies_ok":
            scorre_path = None
            base_shape_path = str(project_root / 'Schematics' / 'shapes' / 'nappies.png')
            base_print_path = str(project_root / 'Schematics' / 'prints' / 'nappies.png')
            recomposed_path = str(project_root / "Reconstructed" / "nappies_ok.png")   
        case "marrone":
            scorre_path = str(project_root / 'dataset_piccoli' / 'Scorre_marrone' / '*.png')
            base_shape_path = str(project_root / 'Schematics' / 'shapes' / 'marrone.png')
            base_print_path = str(project_root / 'Schematics' / 'prints' / 'marrone.png')
            recomposed_path = str(project_root / "Reconstructed" / "marrone.png") 
        case "nappies_misprint":
            scorre_path = str(project_root / 'dataset_medi' / 'Nappi' / '*.png')
            base_shape_path = str(project_root / 'Schematics' / 'shapes' / 'nappies.png')
            base_print_path = str(project_root / 'Schematics' / 'prints' / 'nappies.png')
            recomposed_path = str(project_root / "Reconstructed" / "nappies_misprint.png")
        case _:
            raise ValueError(f"Unknown name: {name}. Please provide a valid name.")
    return scorre_path, base_shape_path, base_print_path, recomposed_path
