{
  pkgs,
  lib,
  config,
  ...
}:
{
  packages = [
    config.languages.python.package.pkgs.pjsua2
    pkgs.zlib
    pkgs.gcc
    pkgs.ninja
    pkgs.python3Packages.freesasa
    (pkgs.texlive.combine {
      inherit (pkgs.texlive)
        scheme-medium
        beamer
        translator
        pgf
        tikz-cd
        xcolor
        booktabs
        siunitx
        collection-fontsrecommended
        latexmk
        biber
        biblatex
        etoolbox
        geometry
        hyperref
        amsmath
        tools
        graphics
        ;
    })
    pkgs.zathura
    pkgs.python313Packages.numpy
    pkgs.python313Packages.matplotlib
    pkgs.python313Packages.spacy
  ];

  languages.python = {
    enable = true;
    poetry = {
      enable = true;
      activate.enable = true;
      install = {
        enable = true;
        installRootPackage = true;
        onlyInstallRootPackage = false;
        verbosity = "more";
      };
    };
  };

  git-hooks.hooks = {
    alejandra.enable = true;
    black.enable = true;
  };

  # ============================================
  # Process Compose Configuration
  # ============================================

  # Process manager settings
  # Define processes
  # process.run.latex-watch = {
  #   exec = "latexmk -pdf -pvc -interaction=nonstopmode presentation.tex";
  #     process-compose = {
  #       working_dir = "./presentation"; # Adjust path
  #       availability = {
  #         restart = "on_failure";
  #         max_restarts = 3;
  #       };
  #       readiness_probe = {
  #         initial_delay_seconds = 2;
  #       };
  #     };
  #   };

  # Python analysis script watcher (example)
  # processes.analysis = {
  #   exec = ''
  #     watchexec -e py -- python scripts/analyze.py
  #   '';
  #   process-compose = {
  #     availability = {
  #       restart = "always";
  #     };
  #   };
  # };

  # Jupyter notebook server (if needed)
  # jupyter = {
  #   exec = "jupyter lab --no-browser --port=8888";
  #   process-compose = {
  #     readiness_probe = {
  #       http_get = {
  #         host = "localhost";
  #         port = 8888;
  #       };
  #       initial_delay_seconds = 5;
  #     };
  #   };
  # };

  # ============================================
  # Scripts (convenience commands)
  # ============================================
  scripts = {
    texbuild.exec = "latexmk -f presentation.tex";
    texclean.exec = "latexmk -c";
    texpurge.exec = "latexmk -C";
    watch.exec = "devenv up"; # Starts process-compose
  };

  enterShell = ''
    echo "Python version: $(python --version)"
    echo "Poetry version: $(poetry --version)"
    echo ""
    echo "📄 LaTeX environment ready"
    echo "   build-pdf      - Single compilation"
    echo "   clean-tex      - Clean auxiliary files"
    echo "   watch          - Start process-compose (latex + viewer)"
    echo "   devenv up      - Same as watch"
    echo ""
  '';
}
