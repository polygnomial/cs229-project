find ~/Sen2Fire -type f | while read -r file; do
  clean_file="${file#"$HOME/Sen2Fire/"}"  # Remove ~/Sen2Fire/ prefix
  wrangler r2 object put --bucket sen2fire --file "$file" --key "$clean_file"
done
