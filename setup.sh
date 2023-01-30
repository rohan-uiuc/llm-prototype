mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
export HUGGINGFACEHUB_API_TOKEN='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'