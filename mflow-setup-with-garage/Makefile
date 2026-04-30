up:
	@if [ ! -f ./garage-bin ]; then \
		echo "Extracting garage binary..."; \
		docker create --name temp-garage dxflrs/garage:v2.3.0; \
		docker cp temp-garage:/garage ./garage-bin; \
		docker rm temp-garage; \
	fi
	docker compose up -d

down:
	docker compose down -v