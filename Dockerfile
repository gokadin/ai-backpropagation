FROM iron/go:dev

WORKDIR /app

ENV SRC_DIR=/go/src/github.com/gokadin/ai-backpropagation

ADD . $SRC_DIR

RUN cd $SRC_DIR; go build -o app; cp app /app/

ENTRYPOINT ["./app"]