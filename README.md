# VSSM — Validação Semântica e Seleção de Modelos

Artefato de pesquisa desenvolvido no âmbito da tese **EDAXs: Abordagem Computacional para Análise de Comentários da Avaliação Institucional na Educação Superior**.

O VSSM apoia a comparação de modelos de análise de sentimentos a partir de comentários textuais e de uma referência humana consolidada. O sistema inclui preparação de dados, normalização de rótulos, comparação de modelos acessados por API e apresentação de indicadores de desempenho.

## Requisitos

- Python 3.11 (recomendado pelo Dockerfile)
- Acesso a pelo menos um provedor de modelos compatível e respectiva chave de API

## Instalação local

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e preencha apenas as chaves dos provedores que utilizar. Nunca publique `.env`.

```bash
python run.py
```

A aplicação inicia por padrão em `http://127.0.0.1:5000`.

## Estrutura

- `app/routes.py`: rotas e interface da aplicação
- `app/services/`: preparação de dados, modelos, métricas e histórico
- `app/templates/` e `app/static/`: interface web
- `data/sample_dataset.csv`: dados de exemplo incluídos no projeto original
- `requirements.txt`: dependências Python
- `Dockerfile` e `docker-compose.yml`: configuração de execução em contêiner

## Privacidade e reprodução

Este repositório não inclui bases institucionais carregadas, histórico de execuções, resultados gerados, chaves de API nem ambientes virtuais. A reprodução dos resultados da tese depende das bases, configurações e condições de acesso aos modelos utilizados no estudo. Serviços externos podem ter custos e limites de utilização.

## Nota

O código mantém algumas referências internas à sigla `SVSM` por compatibilidade com a implementação original. O nome do artefato na tese é **VSSM**. A aplicação deve ser testada no ambiente de destino antes de ser disponibilizada para uso público.
